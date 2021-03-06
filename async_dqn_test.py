import mxnet as mx
import numpy as np
import sym
import argparse
import rl_data
import logging
import os
import threading
import gym
import time
import random
#from tensorboard import summary
#from tensorboard import FileWriter
from datetime import datetime
from collections import deque

T = 0
TMAX = 80000000
t_max = 32

parser = argparse.ArgumentParser(description='Traing A3C with OpenAI Gym')
parser.add_argument('--test', action='store_true',
                    help='run testing', default=False)
parser.add_argument('--log-file', type=str, help='the name of log file')
parser.add_argument('--log-dir', type=str, default="./log",
                    help='directory of the log file')
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str,
                    help='the prefix of the model to save')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")

parser.add_argument('--kv-store', type=str,
                    default='device', help='the kvstore type')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')

parser.add_argument('--num-epochs', type=int, default=120,
                    help='the number of training epochs')
parser.add_argument('--num-examples', type=int, default=1000000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--input-length', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t-max', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.08)

parser.add_argument('--game', type=str, default='Breakout-v0')
parser.add_argument('--num-threads', type=int, default=4)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=100000)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--network-update-frequency', type=int, default=32)
parser.add_argument('--target-network-update-frequency', type=int, default=100)
parser.add_argument('--resized-width', type=int, default=84)
parser.add_argument('--resized-height', type=int, default=84)
parser.add_argument('--agent-history-length', type=int, default=4)
parser.add_argument('--game-source', type=str, default='Gym')
parser.add_argument('--replay-memory-length', type=int, default=32)

args = parser.parse_args()


logdir = args.log_dir
summary_writer = FileWriter(logdir)

def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


def clear_grad(module):
    for grads in module._exec_group.grad_arrays:
        for grad in grads:
            grad -= grad


def asynchronize_network(fromNetwork, toNetwork):
    lock.acquire()
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                fromNetwork._exec_group.grad_arrays]
    for gradsto, gradsfrom in zip(toNetwork._exec_group.grad_arrays,
                                  gradfrom):
        for gradto, gradfrom in zip(gradsto, gradsfrom):
            gradto += gradfrom

    toNetwork.update()
    clear_grad(toNetwork)
    lock.release()


def test_grad(net):
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                net._exec_group.grad_arrays]
    print 'grad', np.sum(gradfrom[6][0].asnumpy(), axis=1)


def copyTargetQNetwork(fromNetwork, toNetwork):

    arg_params, aux_params = fromNetwork.get_params()
    toNetwork.init_params(initializer=None, arg_params=arg_params,
                          aux_params=aux_params, force_init=True)


def load_args():
    model_prefix = args.model_prefix
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix

    if args.load_epoch is not None:
        assert model_prefix is not None
        _, arg_params, aux_params = mx.model.load_checkpoint(
            model_prefix, args.load_epoch)
    else:
        arg_params = aux_params = None
    return arg_params, aux_params


def setup(isGlobal=False):
    '''
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    '''

    #devs = mx.gpu(1)
    devs = mx.cpu()

    arg_params, aux_params = load_args()

    if(args.game_source == 'Gym'):
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.MultiThreadFlappyBirdIter(args.resized_width,
                                                     args.resized_height, args.agent_history_length, visual=True)
    act_dim = dataiter.act_dim

    mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=False), data_names=('data', 'rewardInput', 'actionInput'),
                        label_names=None, context=devs)
    mod.bind(data_shapes=[('data', (args.batch_size, args.agent_history_length,
                                    args.resized_width, args.resized_height)),
                          ('rewardInput', (args.batch_size, 1)),
                          ('actionInput', (args.batch_size, act_dim))],
             label_shapes=None, grad_req='write')

    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)

    if args.load_epoch is not None:
        mod.init_params(arg_params=arg_params, aux_params=aux_params)
    else:
        mod.init_params(initializer)
    # optimizer
    mod.init_optimizer(optimizer='adam',
                       optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 10.0})

    target_mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=True), data_names=('data',),
                               label_names=None, context=devs)

    target_mod.bind(data_shapes=[('data', (1, args.agent_history_length,
                                           args.resized_width, args.resized_height)), ],
                    label_shapes=None, grad_req='null')
    if args.load_epoch is not None:
        target_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    else:
        target_mod.init_params(initializer)
    # optimizer
    target_mod.init_optimizer(optimizer='adam',
                              optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 10.0})
    if(isGlobal == False):
        return mod, target_mod, dataiter
    else:
        return mod, target_mod


def action_select(act_dim, probs, epsilon):
    if(np.random.rand() < epsilon):
        return np.random.choice(act_dim)
    else:
        return np.argmax(probs)


def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]


def actor_learner_thread(thread_id):
    global TMAX, T, Module, Target_module, lock, epoch

    if args.game_source == 'Gym':
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.MultiThreadFlappyBirdIter(args.resized_width,
                                                     args.resized_height, args.agent_history_length, visual=True)
    act_dim = dataiter.act_dim

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    score = np.zeros((args.batch_size, 1))

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.1
    epsilon = 0.1

    t = 0

    s_batch = []
    s1_batch = []
    a_batch = []
    r_batch = []
    R_batch = []
    terminal_batch = []

    # here use replayMemory to fix batch size for training
    replayMemory = []

    while T < TMAX:
        tic = time.time()
        epoch += 1
        terminal = False
        s_t = dataiter.get_initial_state()
        ep_reward = 0
        episode_max_q = 0
        ep_t = 0
        ep_loss = 0
        # perform an episode
        terminal = False
        episode_max_q = 0
        while True:
            # perform n steps
            t_start = t
            s_batch = []
            s1_batch = []
            a_batch = []
            r_batch = []
            R_batch = []
            while not (terminal or ((t - t_start) == args.t_max)):
                # TODO here should be qnet forwarding, not target net. However,
                #       dealing with variable length input in mxnet is not
                #       about one simple api. Needs to change to qnet here.
                batch = mx.io.DataBatch(data=[mx.nd.array([s_t])],
                                        label=None)
                with lock:
                    Target_module.forward(batch, is_train=False)
                    q_out = Target_module.get_outputs()[0].asnumpy()

                # select action using e-greedy
                #print q_out
                action_index = action_select(act_dim, q_out, epsilon)
                #print q_out, action_index

                a_t = np.zeros([act_dim])
                a_t[action_index] = 1

                # scale down eplision
                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / \
                        args.anneal_epsilon_timesteps

                # play one step game
                s_t1, r_t, terminal, info = dataiter.act(action_index)
                r_t = np.clip(r_t, -1, 1)
                t += 1
                T += 1
                ep_t += 1
                ep_reward += r_t
                episode_max_q = max(episode_max_q, np.max(q_out))

                s_batch.append(s_t)
                s1_batch.append(s_t1)
                a_batch.append(a_t)
                r_batch.append(r_t)
                R_batch.append(r_t)
                terminal_batch.append(terminal)
                s_t = s_t1

            if terminal:
                R_t = 0
            else:
                batch = mx.io.DataBatch(data=[mx.nd.array([s_t1])], label=None)
                with lock:
                    Target_module.forward(batch, is_train=False)
                    R_t = np.max(Target_module.get_outputs()[0].asnumpy())

            for i in reversed(range(0, t - t_start)):
                R_t = r_batch[i] + args.gamma * R_t
                R_batch[i] = R_t

            if len(replayMemory) + len(s_batch) > args.replay_memory_length:
                replayMemory[0:(len(s_batch) + len(replayMemory)
                                ) - args.replay_memory_length] = []
            for i in range(0, t - t_start):
                replayMemory.append(
                    (s_batch[i], a_batch[i], r_batch[i], s1_batch[i],
                     R_batch[i],
                     terminal_batch[i]))

            if len(replayMemory) < args.batch_size:
                continue
            minibatch = random.sample(replayMemory, args.batch_size)
            state_batch = ([data[0] for data in minibatch])
            action_batch = ([data[1] for data in minibatch])
            R_batch = ([data[4] for data in minibatch])

            # estimated reward according to target network
            # print mx.nd.array(state_batch), mx.nd.array([R_batch]),
            # mx.nd.array(action_batch)
            batch = mx.io.DataBatch(data=[mx.nd.array(state_batch),
                                          mx.nd.array(np.reshape(
                                              R_batch, (-1, 1))),
                                          mx.nd.array(action_batch)], label=None)

            with lock:
                Module.forward(batch, is_train=True)
                loss = np.mean(Module.get_outputs()[0].asnumpy())
                summary_writer.add_summary(s, T)
                summary_writer.flush()
                Module.backward()
                Module.update()

            if t % args.network_update_frequency == 0 or terminal:
                with lock:
                    copyTargetQNetwork(Module, Target_module)

            if terminal:
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_max_q), "/ EPSILON PROGRESS", t / float(args.anneal_epsilon_timesteps)
                s = summary.scalar('score', ep_reward)
                summary_writer.add_summary(s, T)
                summary_writer.flush()
                elapsed_time = time.time() - start_time
                steps_per_sec = T / elapsed_time
                print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                    T,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
                ep_reward = 0
                episode_max_q = 0
                break

        #print epoch
        if args.save_every != 0 and epoch % args.save_every == 0:
            save_params(args.save_model_prefix, Module, epoch)


def log_config(log_dir=None, log_file=None, prefix=None, rank=0):
    reload(logging)
    head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    if log_dir:
        logging.basicConfig(level=logging.DEBUG, format=head)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not log_file:
            log_file = (prefix if prefix else '') + \
                datetime.now().strftime('_%Y_%m_%d-%H_%M.log')
            #r_t = np.clip(r_t, -1, 1)
            #r_t = np.clip(r_t, -1, 1)
            log_file = log_file.replace('/', '-')
        else:
            log_file = log_file
        log_file_full_name = os.path.join(log_dir, log_file)
        handler = logging.FileHandler(log_file_full_name, mode='w')
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)


def test():
    _, module, dataiter = setup()

    act_dim = dataiter.act_dim
    s_t = dataiter.get_initial_state()
    ep_reward = 0
    while True:
        batch = mx.io.DataBatch(data=[mx.nd.array([s_t])],
                                label=None)
        module.forward(batch, is_train=False)
        q_out = module.get_outputs()[0].asnumpy()
        action_index = np.argmax(q_out)
        a_t = np.zeros([act_dim])
        a_t[action_index] = 1
        s_t1, r_t, terminal, info = dataiter.act(action_index)
        ep_reward += r_t
        if terminal:
            print 'reward', ep_reward
            ep_reward = 0
            s_t1 = dataiter.get_initial_state()
        s_t = s_t1


def train():
    # logging
    np.set_printoptions(precision=3, suppress=True)

    global Module, Target_module, lock, epoch, start_time
    epoch = 0
    Module, Target_module, _ = setup()
    lock = threading.Lock()

    start_time = time.time()
    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(
        thread_id,)) for thread_id in range(args.num_threads)]
    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

    #actor_learner_thread(0)


if __name__ == '__main__':
    if args.test == True:
        test()
    else:
        train()
