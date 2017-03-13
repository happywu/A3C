import mxnet as mx
import numpy as np
import sym
import argparse
import rl_data
import logging
import random
import os
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
parser.add_argument('--batch-size', type=int, default=8)
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
parser.add_argument('--replay-memory-length', type=int, default=5000)
parser.add_argument('--observe-time', type=str, default=100)

args = parser.parse_args()


def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)

def setup():

    devs = mx.cpu()

    dataiter = rl_data.FlappyBirdIter(args.resized_width,
                                      args.resized_height, args.agent_history_length, visual=True)
    act_dim = 2

    mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=False), data_names=('data', 'rewardInput', 'actionInput'),
                        label_names=None, context=devs)
    mod.bind(data_shapes=[('data', (args.batch_size, args.agent_history_length,
                                    args.resized_width, args.resized_height)),
                          ('rewardInput', (args.batch_size, 1)),
                          ('actionInput', (args.batch_size, act_dim))],
             label_shapes=None, grad_req='write')

    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)

    mod.init_params(initializer)
    # optimizer
    mod.init_optimizer(optimizer='adam',
                       optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 10})

    target_mod = mx.mod.Module(sym.get_dqn_symbol(act_dim, ispredict=True), data_names=('data',),
                               label_names=None, context=devs)

    target_mod.bind(data_shapes=[('data', (1, args.agent_history_length,
                                           args.resized_width, args.resized_height)),],
                    label_shapes=None, grad_req='null')
    target_mod.init_params(initializer)
    # optimizer
    target_mod.init_optimizer(optimizer='adam',
                              optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3, 'clip_gradient': 10.0})
    return mod, target_mod, dataiter


def action_select(act_dim, probs, epsilon):
    if(np.random.rand() < epsilon):
        return [np.random.choice(act_dim)]
    else:
        return [np.argmax(probs)]


def sample_final_epsilon():
    final_espilons_ = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_espilons_, 1, p=list(probabilities))[0]


def Train():
    global TMAX, T

    replayMemory = deque()
    module, target_module, dataiter = setup()

    act_dim = dataiter.act_dim

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.2
    epsilon = 0.2

    epoch = 0
    t = 0

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
        while True:
            # Forward q network, get Q(s,a) values
            #print s_t
            temp_a = np.zeros((args.batch_size, act_dim))
            batch = mx.io.DataBatch(data=[mx.nd.array([s_t])], label=None)
            target_module.forward(batch, is_train=False)
            q_out = target_module.get_outputs()[0]

            # select action using e-greedy
            #print 'qvalue', q_out.asnumpy()
            action_index = action_select(act_dim, q_out.asnumpy(), epsilon)
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
            ep_reward += r_t

            replayMemory.append((s_t, a_t, r_t, s_t1, terminal))
            if (len(replayMemory) > args.replay_memory_length):
                replayMemory.popleft()
            s_t = s_t1

            t += 1
            # sample random minibatch and train q network
            if (t> args.observe_time):
                minibatch = random.sample(replayMemory, args.batch_size)
                state_batch = ([data[0] for data in minibatch])
                action_batch =  ([data[1] for data in minibatch])
                reward_batch =  ([data[2] for data in minibatch])
                nextState_batch =  ([data[3] for data in minibatch])
                terminal_batch = ([data[4] for data in minibatch])

                y_batch = np.zeros((args.batch_size,))
                Qvalue = []
                for i in range(args.batch_size):
                    target_module.forward(mx.io.DataBatch(data=[mx.nd.array([nextState_batch[i]])], label=None), is_train=False)
                    Qvalue.append(target_module.get_outputs()[0].asnumpy())
                Qvalue_batch = Qvalue
                y_batch[:] = reward_batch
                for i in range(args.batch_size):
                    if(terminal_batch[i]==False):
                        y_batch[i] += args.gamma * np.max(Qvalue_batch[i], axis=1)
                batch = mx.io.DataBatch(data=[mx.nd.array(state_batch), mx.nd.array(np.array(reward_batch).reshape((-1,1))), mx.nd.array(action_batch)],label=None)
                #print mx.nd.array(state_batch), mx.nd.array(np.array(reward_batch).reshape(-1,1)), mx.nd.array(action_batch)
                module.forward(batch, is_train=True)
                module.backward()
                module.update()
            if (t % args.network_update_frequency):
                arg_params,aux_params=module.get_params() #arg={}
                target_module.init_params(initializer=None, arg_params=arg_params,aux_params=aux_params,force_init=True)
            if (terminal):
                print 'score', ep_reward
                ep_reward = 0
                break

def train():

    Train()


if __name__ == '__main__':
    train()
