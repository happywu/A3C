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

    lock.acquire()
    arg_params, aux_params = fromNetwork.get_params()
    try:
        toNetwork.init_params(initializer=None, arg_params=arg_params,
                              aux_params=aux_params, force_init=True)
    except:
        print 'from ', fromNetwork.get_params()
        print 'to ', toNetwork.get_params()
    lock.release()

def setup():

    devs = mx.gpu(0)

    net = sym.get_symbol_atari(2)
    module = mx.mod.Module(net, data_names=('data','rewardInput'),
                           label_names=None, context=devs)

    module.bind(data_shapes=[('data',(1,1,80,80)),
                             ('rewardInput',(args.batch_size, 1))],
                label_shapes=None,
                grad_req='write')

    module.init_params()
    # optimizer
    module.init_optimizer(kvstore=kv, optimizer='adam',
                        optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return module

def actor_learner_thread(num):
    global TMAX, T
    kv = mx.kvstore.create(args.kv_store)

    module  = setup()

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
            s_batch.append(data)
            print data
            rewardInput = [[0]]
            batch = mx.io.DataBatch(dataa=[data, mx.nd.array(rewardInput)],
                                    label=None)
            module.forward(mx.io.DataBatch(data=batch, label=None), is_train=False)

            policy_log, value_loss, policy_out,value_out = module.get_outputs()
            V.append(value_out.asnumpy())
            probs = policy_out.asnumpy()[0]
            action_index = [np.random.choice(act_dim, p=probs)]

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
            value_out = module.get_outputs()[3]
            R_t = value_out.asnumpy()

        err = 0
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + args.gamma * R_t
            batch = mx.io.DataBatch(data=[s_batch[i],
                                          mx.nd.array(past_rewards[i])],
                                    label=None)

            module.forward(batch, is_train=True)

            advs = np.zeros((1, act_dim))
            advs[:,a_batch[i]] = R_t - V[i]
            advs = mx.nd.array(advs)

            module.backward(out_grads=[advs])

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

    global Qnet, lock
    Qnet, _ = setup()
    lock = threading.Lock()

    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id,)) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

if __name__ == '__main__':
    train()
