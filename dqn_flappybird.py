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
import flappybirdprovider

T = 0
TMAX = 80000000
t_max = 32

parser = argparse.ArgumentParser(description='Traing A3C with OpenAI Gym')
parser.add_argument('--test', action='store_true', help='run testing', default=False)
parser.add_argument('--log-file', type=str, help='the name of log file')
parser.add_argument('--log-dir', type=str, default="./log", help='directory of the log file')
parser.add_argument('--model-prefix', type=str, help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str, help='the prefix of the model to save')
parser.add_argument('--load-epoch', type=int, help="load the model on an epoch using the model-prefix")

parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')

parser.add_argument('--num-epochs', type=int, default=120, help='the number of training epochs')
parser.add_argument('--num-examples', type=int, default=1000000, help='the number of training examples')
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
parser.add_argument('--network-update-frequency', type=int, default=32)
parser.add_argument('--target-network-update-frequency', type=int, default=1000)

args = parser.parse_args()

def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)

def asynchronize_network(fromNetwork, toNetwork):
    lock.acquire()
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                fromNetwork._exec_group.grad_arrays]
    for gradsto, gradsfrom in zip(toNetwork._exec_group.grad_arrays,
                                  gradfrom):
        for gradto, gradfrom in zip(gradsto, gradsfrom):
            gradto += gradfrom
    toNetwork.update()
    lock.release()


def copyTargetQNetwork(fromNetwork, toNetwork):

    arg_params, aux_params = fromNetwork.get_params()
    toNetwork.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params, force_init=True)

def setup():

    '''
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    '''

    devs = mx.cpu()
    act_dim = 2
    network = sym.get_dqn_symbol(act_dim)

    mod = mx.mod.Module(network, data_names=('data','rewardInput','actionInput'),
                        label_names=None,context=devs)
    mod.bind(data_shapes=[('data', (1,1,80,80)),
                          ('rewardInput',(args.batch_size, 1)),
                          ('actionInput', (args.batch_size, act_dim))],
             label_shapes=None, grad_req='write')

    mod.init_params(initializer=mx.init.Xavier(factor_type='in', magnitude=2.34))
    # optimizer
    mod.init_optimizer(optimizer='adam',
                       optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})

    target_network = sym.get_dqn_symbol(act_dim)
    target_mod = mx.mod.Module(target_network, data_names=('data','rewardInput','actionInput'),
                               label_names=None,context=devs)
    target_mod.bind(data_shapes=[('data', (1,1,80,80)),
                                 ('rewardInput',(args.batch_size, 1)),
                                 ('actionInput', (args.batch_size, act_dim))],
                    label_shapes=None, grad_req='write')

    target_mod.init_params(initializer=mx.init.Xavier(factor_type='in',magnitude=2.34))
    # optimizer
    target_mod.init_optimizer(optimizer='adam',
                              optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return mod, target_mod

def action_select(act_dim, probs, epsilon):
    if(np.random.rand()<epsilon):
        return [np.random.choice(act_dim)]
    else:
        return [np.argmax(probs)]

def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]

def actor_learner_thread(thread_id):
    global TMAX, T

    module, target_module = setup()


    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    dataiter = flappybirdprovider.GameDataIter()

    probs_summary_t = 0

    score = np.zeros((args.batch_size, 1))

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0
    epsilon = 0

    epoch = 0
    s_batch = []
    a_batch = []
    y_batch = []

    act_dim = 2

    while T < TMAX:
        tic = time.time()
        t = 0
        t_start = t

        copyTargetQNetwork(Net, module)
        copyTargetQNetwork(Net, target_module)

        epoch += 1
        terminal = False
        s_t = dataiter.state()

        ep_reward = 0
        episode_ave_max_q = 0
        ep_t = 0

        # perform an episode
        while True:
            # Forward q network, get Q(s,a) values
            temp_a = np.zeros((args.batch_size,act_dim))
            batch = mx.io.DataBatch(data=[mx.nd.array(s_t),mx.nd.array([[0]]),
                                          mx.nd.array(temp_a)], label=None)
            module.forward(batch, is_train=False)
            loss, q_out = module.get_outputs()

            print 'prob', q_out.asnumpy()
            # select action using e-greedy
            action_index = action_select(act_dim, q_out, epsilon)
            a_t = np.zeros([act_dim])
            a_t[action_index] = 1

            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / args.anneal_epsilon_timesteps

            # play one step game
            _, r_t, terminal = dataiter.act(action_index)

            # get next state
            s_t1 = dataiter.state()

            # estimated reward according to target network
            batch = mx.io.DataBatch(data=[mx.nd.array(s_t1), mx.nd.array([[0]]),
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


            a_batch.append(a_t.reshape((-1,act_dim)))
            s_batch.append(s_t[0])

            s_t = s_t1
            t += 1
            T += 1

            ep_t += 1
            ep_reward += r_t
            episode_ave_max_q += np.max(q_out.asnumpy())


            batch = mx.io.DataBatch(data=[mx.nd.array(s_t), mx.nd.array([[y_batch[-1]]]), mx.nd.array(a_batch[-1])],
                                    label=None)
            module.forward(batch, is_train=True)
            print module.get_outputs()[0].asnumpy()
            module.backward()

            if t % args.target_network_update_frequency == 0:
                copyTargetQNetwork(Net, target_module)

            if t % args.network_update_frequency == 0 or terminal:
                s_batch = []
                a_batch = []
                y_batch = []

                asynchronize_network(module, Net)
                print 'update!'
                module.update()

            if terminal:
                print "THREAD:", thread_id, "/ TIME", T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", ep_reward, "/ Q_MAX %.4f" % (episode_ave_max_q/float(ep_t)), "/ EPSILON PROGRESS", t/float(args.anneal_epsilon_timesteps)
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
            log_file = (prefix if prefix else '') + datetime.now().strftime('_%Y_%m_%d-%H_%M.log')
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

    global Net, lock
    Net, TargetNet = setup()
    lock = threading.Lock()

    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id,)) for thread_id in range(args.num_threads)]

    '''
    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()
    '''

    actor_learner_thread(0)

if __name__ == '__main__':
    train()