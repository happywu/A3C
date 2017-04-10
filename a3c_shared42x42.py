import mxnet as mx
import numpy as np
import sym
import argparse
import rl_data
import logging
import os
import threading
import gym
import random
'''
from tensorboard import summary
from tensorboard import FileWriter
'''
from datetime import datetime
import time
from a3cmodule import A3CModule
from myinit import normalized_columns_initializer
from myinit import my_conv_initializer

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
parser.add_argument('--batch-size', type=int, default=20)
parser.add_argument('--input-length', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t-max', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.08)

parser.add_argument('--game', type=str, default='Breakout-v0')
parser.add_argument('--num-threads', type=int, default=4)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=100000)
parser.add_argument('--save-every', type=int, default=500)
parser.add_argument('--resized-width', type=int, default=42)
parser.add_argument('--resized-height', type=int, default=42)
parser.add_argument('--agent-history-length', type=int, default=4)
parser.add_argument('--game-source', type=str, default='Gym')
parser.add_argument('--replay-memory-length', type=int, default=32)
args = parser.parse_args()


logdir = args.log_dir
#summary_writer = FileWriter(logdir)

def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


def test_grad(net):
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                net._exec_group.grad_arrays]
    print 'grad', np.sum(gradfrom[6][0].asnumpy(), axis=1)


def getGame():
    if (args.game_source == 'Gym'):
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.MultiThreadFlappyBirdIter(
            args.resized_width, args.resized_height, args.agent_history_length, visual=True)
    return dataiter


def getNet(act_dim):
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    loss_net = sym.get_symbol_atari42x42(act_dim)
    loss_mod = A3CModule(loss_net, data_names=('data', 'rewardInput', 'actionInput', 'tdInput'),
                         label_names=None, context=devs)
    loss_mod.bind(data_shapes=[('data', (args.batch_size,
                                         args.agent_history_length, args.resized_width, args.resized_height)),
                               ('rewardInput', (args.batch_size,
                                                1)),
                               ('actionInput', (args.batch_size,
                                                act_dim)),
                               ('tdInput', (args.batch_size, 1))],
                  label_shapes=None, grad_req='write')

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

    #initializer = mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1)
    #initializer = mx.init.mixed(['fc_value_weight|fc_policy_weight', '.*'],
    #                            [mx.init.uniform(0.001), mx.init.xavier(rnd_type='uniform', factor_type="in", magnitude=1)])
    #initializer = mx.init.Constant(0.0001)
    initializer = mx.init.Mixed(['fc_value', 'fc_policy', '.*'],
                                [normalized_columns_initializer(1.0), normalized_columns_initializer(0.01), my_conv_initializer()])

    if args.load_epoch is not None:
        loss_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    else:
        loss_mod.init_params(initializer=initializer)
    # optimizer
    loss_mod.init_optimizer(optimizer='adam',
                            optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return loss_mod


def action_select(act_dim, probs, epsilon):
    if(np.random.rand() < epsilon):
        return np.random.choice(act_dim)
    else:
        return np.argmax(probs)

def random_choose(act_dim, probs):
    return np.random.choice(act_dim, p=probs)

def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]

def sample_discrete_actions(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.

    Args:
      batch_probs (ndarray): batch of action probabilities BxA
    Returns:
      List consisting of sampled actions
    """
    action_indices = []

    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    batch_probs = batch_probs - np.finfo(np.float32).epsneg

    for i in range(batch_probs.shape[0]):
        histogram = np.random.multinomial(1, batch_probs[i])
        action_indices.append(int(np.nonzero(histogram)[0]))
    return action_indices

def actor_learner_thread(thread_id):
    global TMAX, T, lock, epoch, Module
    #kv = mx.kvstore.create(args.kv_store)

    dataiter = getGame()
    act_dim = dataiter.act_dim
    module = getNet(act_dim)
    module.bind(data_shapes=[('data', (1, args.agent_history_length, args.resized_width, args.resized_height)),
                             ('rewardInput', (1, 1)),
                             ('actionInput', (1, act_dim)),
                             ('tdInput', (1, 1))],
                label_shapes=None, grad_req='null', force_rebind=True)

    act_dim = dataiter.act_dim
    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0
    s_t = dataiter.get_initial_state()
    terminal = False
    # anneal e-greedy probability
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.1
    epsilon = 0.1
    t = 0
    replayMemory = []
    episode_max_p = 0
    max_v = -1000
    while T < TMAX:
        tic = time.time()
        with lock:
            module.copy_param_from_module(Module)
        t_start = t
        epoch += 1
        s_batch = []
        s1_batch = []
        r_batch = []
        a_batch = []
        R_batch = []
        td_batch = []
        V_batch = []
        terminal_batch = []

        while not (terminal or ((t - t_start) == args.t_max)):
            null_r = np.zeros((1, 1))
            null_a = np.zeros((1, act_dim))
            null_td = np.zeros((1, 1))
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t]), mx.nd.array(null_r),
                                          mx.nd.array(null_a), mx.nd.array(null_td)], label=None)

            module.forward(batch, is_train=False)
            policy_out, value_out, total_loss = module.get_outputs()
            probs = policy_out.asnumpy()[0]
            v_t = value_out.asnumpy()
            episode_max_p = max(episode_max_p, max(probs))
            max_v = max(max_v, value_out.asnumpy())
            #print 'prob', probs, 'value',  value_out.asnumpy()
            #print mx.nd.SoftmaxActivation(policy_out2).asnumpy()
            # total_loss.asnumpy(), 'loss_out', loss_out.asnumpy()

            #action_index = action_select(act_dim, probs, epsilon)
            action_index = sample_discrete_actions([probs])[0]
            a_t = np.zeros([act_dim])
            a_t[action_index] = 1
            #print a_t

            '''
            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / \
                           args.anneal_epsilon_timesteps
            '''

            s_t1, r_t, terminal, info = dataiter.act(action_index)
            #with lock:
            #    dataiter.env.render()
            r_t = np.clip(r_t, -1, 1)
            t += 1
            T += 1
            ep_t += 1
            ep_reward += r_t

            s_batch.append(s_t)
            s1_batch.append(s_t1)
            a_batch.append(a_t)
            r_batch.append(r_t)
            V_batch.append(v_t)
            R_batch.append(0)
            td_batch.append(0)
            terminal_batch.append(terminal)
            s_t = s_t1

        if terminal:
            R_t = np.zeros((1, 1))
        else:
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t1]), mx.nd.array(
                null_r), mx.nd.array(null_a)], label=None)
            module.forward(batch, is_train=False)
            #R_t = np.clip(module.get_outputs()[1].asnumpy(), - 2, 2)
            R_t = module.get_outputs()[1].asnumpy()

        for i in reversed(range(0, t - t_start)):
            R_t = r_batch[i] + args.gamma * R_t
            R_batch[i] = R_t
            #print 'R_t!', R_t
            # print mx.nd.array([s_batch[i]]), mx.nd.array(R_t),
            # mx.nd.array([a_batch[i]])
            td_batch[i] = R_t - V_batch[i]
            #print 'adv', td_batch[i], 'R_t', R_t, 'V_t', V_batch[i], 'a_t', a_batch[i]

        if len(replayMemory) + len(s_batch) > args.replay_memory_length:
            replayMemory[0:(len(s_batch) + len(replayMemory)
                            ) - args.replay_memory_length] = []
        for i in range(0, t - t_start):
            replayMemory.append(
                (s_batch[i], a_batch[i], r_batch[i], s1_batch[i],
                 R_batch[i], td_batch[i],
                 terminal_batch[i]))

        if len(replayMemory) < args.batch_size:
            terminal = False
            continue

        minibatch = random.sample(replayMemory, args.batch_size)
        s_minibatch = ([data[0] for data in minibatch])
        a_minibatch = ([data[1] for data in minibatch])
        R_minibatch = ([data[4] for data in minibatch])
        td_minibatch = ([data[5] for data in minibatch])

        batch = mx.io.DataBatch(data=[mx.nd.array(s_minibatch), mx.nd.array(np.array(R_minibatch).reshape(-1,1)),
                                      mx.nd.array(a_minibatch), mx.nd.array(np.array(td_minibatch).reshape(-1, 1))], label=None)

        #print mx.nd.array(s_batch), mx.nd.array(np.array(R_batch).reshape(-1,1)), mx.nd.array(a_batch), mx.nd.array(td_batch)
        with lock:
            Module.clear_gradients()
            Module.forward(batch, is_train=True)
            #print 'loss', np.mean(Module.get_outputs()[2].asnumpy())
            '''
            _, _, total_loss, policy_loss, value_loss = Module.get_outputs()
            s = summary.scalar('total_loss', np.mean(total_loss.asnumpy()))
            summary_writer.add_summary(s, T)
            s = summary.scalar('policy_loss', np.mean(policy_loss.asnumpy()))
            summary_writer.add_summary(s, T)
            s = summary.scalar('value_loss', np.mean(value_loss.asnumpy()))
            summary_writer.add_summary(s, T)
            summary_writer.flush()
            '''

            Module.backward()
            Module.norm_clipping(40.0)
            Module.update()

        if terminal:
            print "THREAD:", thread_id, "/ Epoch", epoch, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ P_MAX %.4f" % episode_max_p, "/ EPSILON PROGRESS", t / float(args.anneal_epsilon_timesteps), "Max_v", max_v
            '''
            s = summary.scalar('score', ep_reward)
            summary_writer.add_summary(s, T)
            summary_writer.flush()
            '''
            # save score to file
            if T % 1000 == 0:
                with open(os.path.join(args.log_dir, 'train_score.txt'), 'a+') as f:
                    print >>f, str(T), str(ep_reward)

            max_v = -1000
            elapsed_time = time.time() - start_time
            steps_per_sec = T / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                T,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
            ep_reward = 0
            episode_max_p = 0
            terminal = False
            s_t = dataiter.get_initial_state()

        if args.save_every != 0 and epoch % args.save_every == 0:
            with lock:
                save_params(args.save_model_prefix, Module, epoch)

def test():
    if args.game_source == 'Gym':
        dataiter = rl_data.GymDataIter(
            args.game, args.resized_width, args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.MultiThreadFlappyBirdIter(
            args.resized_width, args.resized_height, args.agent_history_length)
    act_dim = dataiter.act_dim
    module = getNet(act_dim)

    module.bind(data_shapes=[('data', (1, args.agent_history_length, args.resized_width, args.resized_height)),
                             ('rewardInput', (1, 1)),
                             ('actionInput', (1, act_dim)),
                             ('tdInput', (1, 1))],
                label_shapes=None, grad_req='null', force_rebind=True)
    s_t = dataiter.get_initial_state()

    ep_reward = 0
    while True:
        null_r = np.zeros((args.batch_size, 1))
        null_a = np.zeros((args.batch_size, act_dim))
        null_td = np.zeros((args.batch_size, 1))
        batch = mx.io.DataBatch(data=[mx.nd.array([s_t]), mx.nd.array(null_r),
                                      mx.nd.array(null_a), mx.nd.array(null_td)], label=None)
        module.forward(batch, is_train=False)
        policy_out, value_out, total_loss, loss_out , policy_out2= module.get_outputs()
        probs = policy_out.asnumpy()[0]
        action_index = np.argmax(probs)
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

    kv = mx.kvstore.create(args.kv_store)

    # logging
    np.set_printoptions(precision=3, suppress=True)

    global Module, lock, epoch, start_time
    if args.game_source == 'Gym':
        dataiter = getGame()
        act_dim = dataiter.act_dim
    else:
        act_dim = 2
    epoch = 0
    Module = getNet(act_dim)
    lock = threading.Lock()

    start_time = time.time()
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id,)) for thread_id in range(args.num_threads)]
    if args.num_threads > 1:
        for t in actor_learner_threads:
            t.start()

        for t in actor_learner_threads:
            t.join()
    else:
        actor_learner_thread(0)


if __name__ == '__main__':
    if args.test == True:
        test()
    else:
        train()