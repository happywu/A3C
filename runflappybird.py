import mxnet as mx
import numpy as np
import sym
import argparse
import logging
import os
import threading
from datetime import datetime
import time
import flappybirdprovider

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

    devs = mx.gpu(0)

    net = sym.get_symbol_atari(2)
    module = mx.mod.Module(net, data_names=['data'],
                           label_names=(['policy_label', 'value_label']), context=devs)

    module.bind(data_shapes=[('data',(1,1,80,80))],
                label_shapes=[('policy_label', (1, )),
                              ('value_label', (1, 1))],
                grad_req='add')

    return module

def actor_learner_thread(num):
    global TMAX, T
    kv = mx.kvstore.create(args.kv_store)

    module = setup()

    module.init_params()
    # optimizer
    module.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})

    copyTargetQNetwork(Qnet, module)

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    gamedata = flappybirdprovider.GameDataIter()
    terminal = False
    s_t = gamedata.state()

    score = np.zeros((args.batch_size, 1))
    act_dim = 2

    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []
        t = 0
        t_start = t

        V = []
        while not (terminal or ((t - t_start)  == args.t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            data = gamedata.data()

            print data
            module.forward(mx.io.DataBatch(data=data, label=None), is_train=False)
            probs, _, val = module.get_outputs()
            V.append(val.asnumpy())
            probs = probs.asnumpy()[0]
            action_index = [np.random.choice(act_dim, p=probs)]

            s_batch.append(data)
            a_batch.append(action_index)

            _, r_t, terminal = gamedata.act(action_index)
            ep_reward += r_t

            past_rewards.append(r_t.reshape((-1, 1)))

            t += 1
            T += 1
            ep_t += 1

        if terminal:
            R_t = np.zeros((1,1))
        else:
            _, _, val = module.get_outputs()
            R_t = val.asnumpy()

        err = 0
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + args.gamma * R_t
            adv =  np.tile(R_t - V[i], (1, act_dim))

            batch = mx.io.DataBatch(data=s_batch[i],
                                    label=[mx.nd.array(a_batch[i]), mx.nd.array(R_t)])

            module.forward(batch, is_train=True)

            pi = module.get_outputs()[1]

            h = args.beta * (mx.nd.log(pi+1e-6)+1)

            module.backward([mx.nd.array(adv), h])

            err += (adv**2).mean()
            score += past_rewards[i]
        module.update()
        copyTargetQNetwork(module, Qnet)

        if terminal:
            print 'Thread, ', num, 'Eposide end! reward ', ep_reward, T
            ep_reward = 0
            terminal = False

def train():
    kv = mx.kvstore.create(args.kv_store)

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
