import mxnet as mx
import numpy as np
import sym
import argparse
import rl_data
import logging
import os
import threading
import gym
from datetime import datetime
import time
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
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--input-length', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t-max', type=int, default=4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.08)

parser.add_argument('--game', type=str, default='Breakout-v0')
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=100000)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--network-update-frequency', type=int, default=32)
parser.add_argument('--target-network-update-frequency', type=int, default=100)
parser.add_argument('--resized-width', type=int, default=84)
parser.add_argument('--resized-height', type=int, default=84)
parser.add_argument('--agent-history-length', type=int, default=4)
parser.add_argument('--game-source', type=str, default='Gym')

args = parser.parse_args()


def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


def clear_grad(module):
    for grads in module._exec_group.grad_arrays:
        for grad in grads:
            grad -= grad
    '''
    for i in range(len(module._exec_group.grad_arrays)):
        for j in range(len(module._exec_group.grad_arrays[i])):
            #module._exec_group.grad_arrays[i][j] = mx.nd.zeros(module._exec_group.grad_arrays[i][j].shape)
            module._exec_group.grad_arrays[i][j] -= module._exec_group.grad_arrays[i][j]
    '''


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


def setup(isGlobal=False):
    '''
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    '''

    #devs = mx.gpu(1)
    devs = mx.cpu()

    if(args.game_source == 'Gym'):
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.FlappyBirdIter(args.resized_width,
                                          args.resized_height, args.agent_history_length, visual=True)
    act_dim = dataiter.act_dim

    mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=False), data_names=('data', 'rewardInput', 'actionInput'),
                        label_names=None, context=devs)
    mod.bind(data_shapes=[('data', (args.batch_size, args.agent_history_length,
                                    args.resized_width, args.resized_height)),
                          ('rewardInput', (args.batch_size, 1)),
                          ('actionInput', (args.batch_size, act_dim))],
             label_shapes=None, grad_req='add')

    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)

    mod.init_params(initializer)
    # optimizer
    mod.init_optimizer(optimizer='adam',
                       optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 1.0})

    target_mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=True), data_names=('data',),
                               label_names=None, context=devs)

    target_mod.bind(data_shapes=[('data', (1, args.agent_history_length,
                                           args.resized_width, args.resized_height)), ],
                    label_shapes=None, grad_req='add')
    target_mod.init_params(initializer)
    # optimizer
    target_mod.init_optimizer(optimizer='adam',
                              optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 1.0})
    if(isGlobal == False):
        return mod, target_mod, dataiter
    else:
        return mod, target_mod


def action_select(act_dim, probs, epsilon):
    if(np.random.rand() < epsilon):
        return [np.random.choice(act_dim)]
    else:
        return [np.argmax(probs)]


def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]


def actor_learner_thread(thread_id):
    global TMAX, T, Module, Target_module, lock

    if args.game_source == 'Gym':
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.FlappyBirdIter(args.resized_width,
                                          args.resized_height, args.agent_history_length, visual=True)
    act_dim = dataiter.act_dim

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    probs_summary_t = 0

    score = np.zeros((args.batch_size, 1))

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.2
    epsilon = 0.2

    epoch = 0
    t = 0

    s_batch = []
    s1_batch = []
    a_batch = []
    r_batch = []
    R_batch = []
    terminal_batch = []

    # here use replayMemory to fix batch size for training
    replayMemory = deque()

    while T < TMAX:
        tic = time.time()
        epoch += 1
        terminal = False
        s_t = dataiter.get_initial_state()
        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0
        ep_loss = 0
        # perform an episode
        terminal = False
        while True:
            # perform n steps
            t_start = t
            while not (terminal or ((t - t_start) == args.t_max)):
                batch = mx.io.DataBatch(data=[mx.nd.array([s_t]), mx.nd.array(np.zeros(
                    args.batch_size, 1), mx.nd.array(np.zeros(args.batch_size, act_dim)))], label=None)

                Module.forward(batch, is_train=False)
                q_out = Module.get_outputs()[1].asnumpy()

                # select action using e-greedy
                action_index = action_select(act_dim, q_out, epsilon)
                #print q_out.asnumpy(), action_index

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

                s_batch.append(s_t)
                s1_batch.append(s_t1)
                a_batch.append(a_t)
                r_batch.append(r_t)
                R_batch.append(r_t)
                terminal_batch.append(terminal)


            if terminal:
                R_t = 0
            else:
                batch = mx.io.DataBatch(data=[mx.nd.array([s_t1])], label=None)
                Target_module.forward(batch, is_train=False)
                R_t = np.max(Target_module.get_outputs()[0].asnumpy())

            for i in reversed(range(t_start, t)):
                R_t = r_batch[i] + args.gamma * R_t
                R_batch[i] = R_t
            for i in range((t_start, t)):
                replayMemory.append((s_batch[i], a_batch[i], r_batch[i], s1_batch[i], R_batch[i], terminal_batch[i]))
                if len(replayMemory) > args.replay_memory_length:
                    replayMemory.popleft()
            
            # estimated reward according to target network
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t1]), mx.nd.array([[0]]),
                                          mx.nd.array(temp_a)], label=None)
            #print s_t1[0], mx.nd.array([[0]]), mx.nd.array(temp_a)

            target_module.forward(batch, is_train=False)
            target_loss, target_q_out = target_module.get_outputs()
            clipped_r_t = np.clip(r_t, -1, 1)
            if terminal:
                y_batch.append(clipped_r_t)
            else:
                y_batch.append(clipped_r_t + args.gamma *
                               np.max(target_q_out.asnumpy()))

            a_batch.append(a_t.reshape((-1, act_dim)))
            s_batch.append(s_t[0])

            s_t = s_t1
            t += 1
            T += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(q_out.asnumpy())

            batch = mx.io.DataBatch(data=[mx.nd.array([s_t]),
                                          mx.nd.array([[y_batch[-1]]]), mx.nd.array(a_batch[-1])],
                                    label=None)
            module.forward(batch, is_train=True)
            ep_loss += module.get_outputs()[0].asnumpy()
            module.backward()

            if t % args.target_network_update_frequency == 0:
                copyTargetQNetwork(Net, TargetNet)
                copyTargetQNetwork(Net, target_module)

            if t % args.network_update_frequency == 0 or terminal:
                s_batch = []
                a_batch = []
                y_batch = []

                asynchronize_network(module, Net)
                #test_grad(Net)
                module.update()
                #param = module.get_params()
                #print 'loss', ep_loss
                #print 'gradient', test_grad(module)
                #print "module", np.sum(param[0]['qvalue_weight'].asnumpy(), axis=1)
                #param = Net.get_params()
                #print 'Net', np.sum(param[0]['qvalue_weight'].asnumpy(), axis=1)
                #print 'weight', param['arg_params']['qvalue_weight'].asnumpy()
                ep_loss = 0

                '''
                gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                            module._exec_group.grad_arrays]
                print 'before update, grad', gradfrom[1][0].asnumpy()
                '''
                clear_grad(module)

            if terminal:
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q / float(ep_t)), "/ EPSILON PROGRESS", t / float(args.anneal_epsilon_timesteps)
                break

        if args.save_every != 0 and epoch % args.save_every == 0:
            save_params(args.save_model_prefix, Net, epoch)


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


def train():
    # logging
    np.set_printoptions(precision=3, suppress=True)

    Module, Target_module, _ = setup()
    lock = threading.Lock()

    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(
        thread_id,)) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()
    #actor_learner_thread(0)

if __name__ == '__main__':
    train()
