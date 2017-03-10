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
parser.add_argument('--resized-width', type=int, default=84)
parser.add_argument('--resized-height', type=int, default=84)
parser.add_argument('--agent-history-length', type=int, default=4)

parser.add_argument('--game-source', type=str, default='Gym')

args = parser.parse_args()

def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)

def copyTargetQNetwork(fromNetwork, toNetwork):
    lock.acquire()
    gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                fromNetwork._exec_group.grad_arrays]
    for gradsto, gradsfrom in zip(toNetwork._exec_group.grad_arrays,
                                  gradfrom):
        for gradto, gradfrom in zip(gradsto, gradsfrom):
            gradto += gradfrom
    toNetwork.update()
    lock.release()

def setup():

    '''
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]
    '''

    #devs = mx.gpu(0)
    devs = mx.cpu()
    if (args.game_source=='Gym'):
        dataiter = rl_data.GymDataIter(args.game, args.resized_width,
                                       args.resized_height, args.agent_history_length)
    else:
        dataiter = rl_data.FlappyBirdIter(args.resized_width, args.resized_height, args.agent_history_length, visual=True)

    act_dim = dataiter.act_dim
    loss_net = sym.get_symbol_atari(act_dim)

    loss_mod = mx.mod.Module(loss_net, data_names=('data','rewardInput','actionInput'),
                             label_names=None,context=devs)
    loss_mod.bind(data_shapes=[('data', (args.batch_size,
        args.agent_history_length, args.resized_width, args.resized_height)),
                               ('rewardInput',(args.batch_size,
                                    1)),
                               ('actionInput', (args.batch_size,
                                    act_dim))],
                  label_shapes=None, grad_req='add')

    model_prefix = args.model_prefix
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix

    if args.load_epoch is not None:
        assert model_prefix is not None
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    else:
        arg_params = aux_params = None

    loss_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # optimizer
    loss_mod.init_optimizer(optimizer='adam',
                       optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return loss_mod , dataiter

def action_select(act_dim, probs, epsilon):
    if(np.random.rand()<epsilon):
        return [np.random.choice(act_dim)]
    else:
        return [np.argmax(probs)]

def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]

def actor_learner_thread(num):
    global TMAX, T
    #kv = mx.kvstore.create(args.kv_store)

    module, dataiter = setup()

    copyTargetQNetwork(Net, module)

    act_dim = dataiter.act_dim

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    probs_summary_t = 0

    s_t = dataiter.get_initial_state()
    terminal = False

    score = np.zeros((args.batch_size, 1))

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0

    epoch = 0
    while T < TMAX:
        tic = time.time()
        s_batch = []
        past_rewards = []
        a_batch = []
        t = 0
        t_start = t
        copyTargetQNetwork(Net, module)
        V = []
        epoch += 1
        print 'start!'
        while not (terminal or ((t - t_start)  == args.t_max)):
            s_batch.append(s_t)
            temp_a = np.zeros((args.batch_size,act_dim))

            batch = mx.io.DataBatch(data=[mx.nd.array([s_t]),mx.nd.array([[0]]),
                mx.nd.array(temp_a)], label=None)

            module.forward(batch, is_train=False)

            policy_out, value_out, total_loss = module.get_outputs()
            V.append(value_out.asnumpy())
            probs = policy_out.asnumpy()[0]
            print probs

            action_index = action_select(act_dim, probs, epsilon)
            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / args.anneal_epsilon_timesteps


            s_t1, r_t, terminal, info = dataiter.act(action_index)

            action_index1 = np.zeros(act_dim)
            action_index1[action_index]=1
            a_batch.append(mx.io.array(action_index1.reshape((-1,act_dim))))
            ep_reward += r_t

            past_rewards.append(r_t)

            t += 1
            T += 1
            ep_t += 1
            s_t = s_t1

        if terminal:
            R_t = np.zeros((1,1))
        else:
            value_out = module.get_outputs()[1]
            R_t = value_out.asnumpy()

        err = 0
        R_batch = []
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + args.gamma * R_t
            R_batch.append(mx.io.array(R_t))
            score += past_rewards[i]
            batch = mx.io.DataBatch(data=[mx.nd.array([s_batch[i]]), mx.io.array(R_t),
                a_batch[i]], label=None)
            module.forward(batch, is_train=True)
            module.backward()

        copyTargetQNetwork(module, Net)
        module.update()

        logging.info('fps: %f err: %f score: %f T: %f'%(args.batch_size/(time.time()-tic), err/args.t_max, score.mean(), T))

        if terminal:
            print 'Thread, ', num, 'Eposide end! reward ', ep_reward, T
            ep_reward = 0
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

    kv = mx.kvstore.create(args.kv_store)

    # logging
    np.set_printoptions(precision=3, suppress=True)

    global Net, lock
    Net, _ = setup()
    lock = threading.Lock()

    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
        args=(thread_id,)) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

        #actor_learner_thread(0)

if __name__ == '__main__':
    train()

