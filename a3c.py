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
from a3cmodule import A3CModule

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
parser.add_argument('--num-threads', type=int, default=3)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=100000)
parser.add_argument('--save-every', type=int, default=1000)
parser.add_argument('--resized-width', type=int, default=84)
parser.add_argument('--resized-height', type=int, default=84)
parser.add_argument('--agent-history-length', type=int, default=4)
parser.add_argument('--game-source', type=str, default='Gym')

args = parser.parse_args()


def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)


def test_grad(net):
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                net._exec_group.grad_arrays]
    print 'grad', gradfrom[7][0].asnumpy()


def getGame():
    if (args.game_source == 'Gym'):
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.MultiThreadFlappyBirdIter(
            args.resized_width, args.resized_height, args.agent_history_length, visual=True)
    return dataiter


def setup(act_dim):
    '''
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    '''
    #devs = mx.gpu(0)
    devs = mx.cpu()
    loss_net = sym.get_symbol_atari(act_dim)
    loss_mod = A3CModule(loss_net, data_names=('data', 'rewardInput', 'actionInput'),
                         label_names=None, context=devs)
    loss_mod.bind(data_shapes=[('data', (args.batch_size,
                                         args.agent_history_length, args.resized_width, args.resized_height)),
                               ('rewardInput', (args.batch_size,
                                                1)),
                               ('actionInput', (args.batch_size,
                                                act_dim))],
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

    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)
    if args.load_epoch is not None:
        loss_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    else:
        loss_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # optimizer
    loss_mod.init_optimizer(optimizer='adam',
                            optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 0.001})
    return loss_mod


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
    global TMAX, T, lock, epoch
    #kv = mx.kvstore.create(args.kv_store)

    dataiter = getGame()
    act_dim = dataiter.act_dim
    module = setup(act_dim)
    module.bind(data_shapes=[('data', (args.batch_size,
                                       args.agent_history_length, args.resized_width, args.resized_height)),
                             ('rewardInput', (args.batch_size,
                                              1)),
                             ('actionInput', (args.batch_size,
                                              act_dim))],
                label_shapes=None, grad_req='add', force_rebind=True)

    act_dim = dataiter.act_dim
    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0
    s_t = dataiter.get_initial_state()
    terminal = False
    # anneal e-greedy probability
    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1
    epsilon = 1
    t = 0
    while T < TMAX:
        tic = time.time()
        with lock:
            module.copy_from_module(Net)
        module.clear_gradients()
        t_start = t
        epoch += 1
        s_batch = []
        s1_batch = []
        r_batch = []
        a_batch = []
        R_batch = []
        terminal_batch = []
        episode_max_p = 0

        while not (terminal or ((t - t_start) == args.t_max)):
            null_r = np.zeros((args.batch_size, 1))
            null_a = np.zeros((args.batch_size, act_dim))
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t]), mx.nd.array(null_r),
                                          mx.nd.array(null_a)], label=None)

            module.forward(batch, is_train=False)
            policy_out, value_out, total_loss, loss_out = module.get_outputs()
            probs = policy_out.asnumpy()[0]
            episode_max_p = max(episode_max_p, max(probs))
            # print 'prob', probs, 'value',  value_out.asnumpy(), 'loss',
            # total_loss.asnumpy(), 'loss_out', loss_out.asnumpy()

            action_index = action_select(act_dim, probs, epsilon)
            a_t = np.zeros([act_dim])
            a_t[action_index] = 1

            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / \
                    args.anneal_epsilon_timesteps

            s_t1, r_t, terminal, info = dataiter.act(action_index)
            r_t = np.clip(r_t, -1, 1)
            t += 1
            T += 1
            ep_t += 1
            ep_reward += r_t

            s_batch.append(s_t)
            s1_batch.append(s_t1)
            a_batch.append(a_t)
            r_batch.append(r_t)
            R_batch.append(r_t)
            terminal_batch.append(terminal)
            s_t = s_t1

        if terminal:
            R_t = np.zeros((1, 1))
        else:
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t1]), mx.nd.array(
                null_r), mx.nd.array(null_a)], label=None)
            module.forward(batch, is_train=False)
            R_t = module.get_outputs()[1].asnumpy()

        module.clear_gradients()
        for i in reversed(range(0, t - t_start)):
            R_t = r_batch[i] + args.gamma * R_t
            R_batch[i] = R_t
            print 'R_t!', R_t
            # print mx.nd.array([s_batch[i]]), mx.nd.array(R_t),
            # mx.nd.array([a_batch[i]])
            batch = mx.io.DataBatch(data=[mx.nd.array([s_batch[i]]), mx.nd.array(R_t),
                                          mx.nd.array([a_batch[i]])], label=None)
            #print 'train! ', 'R_t', R_t, 'a_t', a_batch[i]
            module.forward(batch, is_train=True)
            print 'loss', module.get_outputs()[2].asnumpy(), 'value', module.get_outputs()[1].asnumpy()
            module.backward()
            module.clip_gradients(10)

        #print t, t_start, len(s_batch), len(R_batch), len(a_batch)
        # print mx.nd.array(s_batch), mx.nd.array(np.reshape(R_batch,(-1, 1))),
        # mx.nd.array(a_batch)
        with lock:
            Net.add_gradients_from_module(module)
            Net.clip_gradients(10)
            Net.update()

        module.update()
        module.clear_gradients()

        if terminal:
            print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ P_MAX %.4f" % episode_max_p, "/ EPSILON PROGRESS", t / float(args.anneal_epsilon_timesteps)
            ep_reward = 0
            episode_max_p = 0
            terminal = False
            s_t = dataiter.get_initial_state()

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

    kv = mx.kvstore.create(args.kv_store)

    # logging
    np.set_printoptions(precision=3, suppress=True)

    global Net, lock, epoch
    #dataiter = getGame()
    epoch = 0

    act_dim = 2
    Net = setup(act_dim)
    lock = threading.Lock()

    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id,)) for thread_id in range(args.num_threads)]
    '''
    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()
    '''

    actor_learner_thread(0)


if __name__ == '__main__':
    train()
