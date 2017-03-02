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
TMAX = 8000000
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

args = parser.parse_args()

def copyTargetQNetwork(fromNetwork, toNetwork):
    time_now = time.time()
    arg_params, aux_params = fromNetwork.get_params()

    try: 
        toNetwork.init_params(initializer=None, arg_params=arg_params,
                aux_params=aux_params, force_init=True)
    except:
        print 'from ', fromNetwork.get_params()
        print 'to ', toNetwork.get_params()

def setup():

    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    dataiter = rl_data.GymDataIter(args.game, args.input_length, web_viz=False)
    act_dim = dataiter.act_dim
    net = sym.get_symbol_atari(act_dim)
    module = mx.mod.Module(net, data_names=[d[0] for d in
            dataiter.provide_data],
            label_names=(['policy_label', 'value_label']), context=devs)

    '''
    module = mx.mod.Module(net, data_names=('data'),
            label_names=(['policy_label', 'value_label']), context=devs)
    '''
    '''
    print dataiter.provide_data
    module.bind(data_shapes=[('data',(args.batch_size,12, 210, 160))], label_shapes=[('policy_label',
        (args.batch_size,)), ('value_label', (args.batch_size, 1))],
        grad_req='add')
    '''

    module.bind(data_shapes=dataiter.provide_data,
                label_shapes=[('policy_label', (args.batch_size, )),
                    ('value_label', (args.batch_size, 1))],
                grad_req='add')
    

    return module, dataiter


def actor_learner_thread(num):
    global TMAX, T
    kv = mx.kvstore.create(args.kv_store)

    module, dataiter = setup()

    module.init_params()
    # optimizer
    module.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})

    copyTargetQNetwork(Qnet, module)
    act_dim = dataiter.act_dim

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    probs_summary_t = 0

    #s_t = env.get_initial_state()
    dataiter.reset()
    terminal = False
    s_t = dataiter.data()

    score = np.zeros((args.batch_size, 1))
    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []
        t = 0
        t_start = t

        V = []
        while not (terminal or ((t - t_start)  == args.t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            data = dataiter.data()

            #print 'forward1 ', data,  '\n'
            module.forward(mx.io.DataBatch(data=data, label=None), is_train=False)
            probs, _, val = module.get_outputs()
            V.append(val.asnumpy())
            probs = probs.asnumpy()[0]
            action_index = [np.random.choice(act_dim, p=probs)]

            #if probs_summary_t % 1000 == 0:
            #    print "Prob, Val", np.max(probs), "V ", val.asnumpy()

            s_batch.append(data)
            a_batch.append(action_index)

            r_t, terminal = dataiter.act(action_index)
            ep_reward += r_t

            past_rewards.append(r_t.reshape((-1, 1)))

            t += 1
            T += 1
            ep_t += 1
            probs_summary_t += 1

        if terminal:
            R_t = np.zeros((1,1))
        else:
            _, _, val = module.get_outputs()
            R_t = val.asnumpy()

        R_batch = np.zeros(t)
        err = 0
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + args.gamma * R_t
            adv =  np.tile(R_t - V[i], (1, act_dim))
            #print 'adv', adv

            #print mx.nd.array(a_batch[i])
            batch = mx.io.DataBatch(data=s_batch[i],
                    label=[mx.nd.array(a_batch[i]), mx.nd.array(R_t)])

            print 'forward2 ', s_batch[i], mx.nd.array(a_batch[i]), mx.nd.array(R_t)
            module.forward(batch, is_train=True)

            pi = module.get_outputs()[1]

            h = args.beta * (mx.nd.log(pi+1e-6)+1)
            
            #print 'gradient: ',adv, h.asnumpy()
            module.backward([mx.nd.array(adv), h])

            err += (adv**2).mean()
            score += past_rewards[i]
            #if T % 100 == 0 : 
                #print 'pi ', pi.asnumpy()
                #print 'h ', h.asnumpy()
                #print 'T ', T
                #print 'err ', err
        module.update()
        copyTargetQNetwork(module, Qnet)
        #if T % 1000 == 0 :
        #    print err/args.t_max, score.mean(), T

        if terminal:
            print 'Thread, ', num, 'Eposide end! reward ', ep_reward, T
            ep_reward = 0
            terminal = False
            dataiter.reset()


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

    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix

    if args.load_epoch is not None:
        assert model_prefix is not None
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    else:
        arg_params = aux_params = None

    # logging
    np.set_printoptions(precision=3, suppress=True)

    global Qnet
    Qnet, _ = setup()
    Qnet.init_params()
    # optimizer
    Qnet.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})

    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id,)) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

if __name__ == '__main__':
    train()
