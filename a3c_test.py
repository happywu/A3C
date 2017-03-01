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

def setup():

    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    #dataiters = [rl_data.GymDataIter(args.game, args.input_length, web_viz=False) for _ in range(args.num_threads)]
    dataiter = rl_data.GymDataIter(args.game, args.input_length, web_viz=False) 
    act_dim = dataiter.act_dim
    net = sym.get_symbol_atari(act_dim)
    module = mx.mod.Module(net, data_names=[d[0] for d in
                                            dataiter.provide_data],
                           label_names=(['policy_label', 'value_label']), context=devs)

    module.bind(data_shapes=dataiter.provide_data,
                label_shapes=[('policy_label', (args.batch_size, )),
                              ('value_label', (args.batch_size, 1))],
                grad_req='add')

    return module, dataiter

def actor_learner_thread(num):
    kv = mx.kvstore.create(args.kv_store)

    module, dataiter = setup()

    module.init_params()
    module.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    # logging
    np.set_printoptions(precision=3, suppress=True)
    act_dim = dataiter.act_dim
    time.sleep(2*num)
    dataiter.reset()

    for _ in range(100):
        data = dataiter.data()
        module.forward(mx.io.DataBatch(data=data, label=None), is_train=False)
        action_index = np.random.choice(act_dim)
        dataiter.act([action_index])
        print '%d\n' % num
        print data



def train():
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
        args=(thread_id, )) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

if __name__ == '__main__':
    train()
