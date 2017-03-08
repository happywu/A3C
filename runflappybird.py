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
parser.add_argument('--model-prefix', type=str, default='flappybird', help='the prefix of the model to load')
parser.add_argument('--save-model-prefix', type=str, default='flappybird', help='the prefix of the model to save')
parser.add_argument('--load-epoch', type=int, default=0, help="load the model on an epoch using the model-prefix")

parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')

parser.add_argument('--num-epochs', type=int, default=120, help='the number of training epochs')
parser.add_argument('--num-examples', type=int, default=1000000, help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--input-length', type=int, default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t-max', type=int, default=16)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--beta', type=float, default=0.08)

parser.add_argument('--game', type=str, default='Breakout-v0')
parser.add_argument('--num-threads', type=int, default=3)
parser.add_argument('--epsilon', type=float, default=1)
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=1000000)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--load-model', action='store_true', default=False)



args = parser.parse_args()

def save_params(save_pre, model, epoch):
    model.save_checkpoint(save_pre, epoch, save_optimizer_states=True)

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
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu()

    net = sym.get_symbol_atari(2)
    module = mx.mod.Module(net, data_names=('data','rewardInput'),
                           label_names=None, context=devs)

    module.bind(data_shapes=[('data',(1,1,80,80)),
                             ('rewardInput',(args.batch_size, 1))],
                label_shapes=None,
                grad_req='add')
    module.init_params()

    if args.load_epoch !=0 :
        load_sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
        module.set_params(arg_params=arg_params, aux_params=aux_params)
        print 'True'

    module.init_optimizer(kvstore=kv, optimizer='adam',
                optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return module
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

    np.random.seed()
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

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 1.0
    epsilon = 1.0
    epoch = 0

    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []
        t = 0
        t_start = t
        V = []
        Done = []
        epoch += 1
        while not (terminal or ((t - t_start)  == args.t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            data = gamedata.state()
            s_batch.append(data)
            rewardInput = [[0]]
            batch = mx.io.DataBatch(data=[mx.nd.array(data), mx.nd.array(rewardInput)],
                                    label=None)
            module.forward(batch, is_train=False)

            policy_log, value_loss, policy_out,value_out = module.get_outputs()

            #print policy_log.asnumpy(), value_loss.asnumpy(), policy_out.asnumpy(), value_out.asnumpy()
            V.append(value_out.asnumpy())
            probs = policy_out.asnumpy()[0]

            action_index = action_select(act_dim, probs, epsilon)

            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / args.anneal_epsilon_timesteps
            a_batch.append(action_index)

            _, r_t, terminal = gamedata.act(action_index)
            if r_t>0 and t % 5 != 0:
                r_t = 0
            r_t = np.array([r_t])
            ep_reward += r_t
            Done.append(terminal)

            past_rewards.append(r_t.reshape((-1, 1)))

            t += 1
            T += 1
            ep_t += 1

        if terminal:
            R_t = np.array([-0.1])
        else:
            value_out = module.get_outputs()[3]
            R_t = value_out.asnumpy()

        err = 0
        print R_t
        for i in reversed(range(t_start, t-1)):
            R_t = past_rewards[i] + args.gamma * R_t
            batch = mx.io.DataBatch(data=[mx.nd.array(s_batch[i]),
                                          mx.nd.array(R_t)],
                                    label=None)


            module.forward(batch, is_train=True)
            #print past_rewards[i], module.get_outputs()[3].asnumpy(), 'value_loss', module.get_outputs()[1].asnumpy(), 'log_policy', module.get_outputs()[0].asnumpy(), 'policy_out', module.get_outputs()[2].asnumpy()
            print 'value_loss', module.get_outputs()[1].asnumpy(), 'value_out', module.get_outputs()[3].asnumpy(), R_t, past_rewards[i], 'policy_out', module.get_outputs()[2].asnumpy()
            advs = np.zeros((1, act_dim))
            advs[:,a_batch[i]] = (R_t - V[i])
            advs = mx.nd.array(advs)
            #print 'advs ', R_t, V[i], advs.asnumpy()
            module.backward(out_grads=[advs])

            #err += (adv**2).mean()
            score += past_rewards[i]

        module.update()
        copyTargetQNetwork(module, Qnet)

        if terminal:
            print 'Thread, ', num, 'Eposide end! reward ', ep_reward, T
            ep_reward = 0
            terminal = False

        if args.save_every !=0 and epoch % args.save_every == 0:
            save_params(args.save_model_prefix, Qnet, epoch)

def train():
    global Qnet, lock
    Qnet = setup()
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
