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
parser.add_argument('--save-model-prefix', type=str, default='flappybird2', help='the prefix of the model to save')
parser.add_argument('--load-epoch', type=int, help="load the model on an epoch using the model-prefix")

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
parser.add_argument('--anneal-epsilon-timesteps', type=int, default=10000)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--load-model', action='store_true', default=False)


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

    devs = mx.cpu()

    act_dim = 2
    loss_net = sym.get_symbol_atari(act_dim)

    loss_mod = mx.mod.Module(loss_net, data_names=('data','rewardInput','actionInput'),
                             label_names=None,context=devs)
    loss_mod.bind(data_shapes=[('data',(1,1,80,80)),
                               ('rewardInput',(args.batch_size, 1)),
                               ('actionInput', (args.batch_size, act_dim))],
                  label_shapes=None, grad_req='write')

    model_prefix = args.model_prefix
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix

    if args.load_epoch is not None:
        assert model_prefix is not None
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
        print 'load!'
    else:
        arg_params = aux_params = None

    #mod.init_params(arg_params=arg_params, aux_params=aux_params)

    loss_mod.init_params(arg_params=arg_params, aux_params=aux_params)
    # optimizer
    loss_mod.init_optimizer(optimizer='adam',
                            optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    return loss_mod

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

    copyTargetQNetwork(Net, module)

    # Set up per-episode counters
    ep_reward = 0
    ep_t = 0

    gamedata = flappybirdprovider.GameDataIter()
    terminal = False
    s_t = gamedata.state()

    score = np.zeros((args.batch_size, 1))
    act_dim = 2

    final_epsilon = sample_final_epsilon()
    initial_epsilon = 0.1
    epsilon = 0.1
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
            batch = mx.io.DataBatch(data=[mx.nd.array(data),
                                          mx.nd.array([[0]]), mx.nd.array([[0, 1]])],
                                    label=None)
            module.forward(batch, is_train=False)

            policy_out, value_out , total_loss = module.get_outputs()
            #print policy_log.asnumpy(), value_loss.asnumpy(), policy_out.asnumpy(), value_out.asnumpy()
            V.append(value_out.asnumpy())
            probs = policy_out.asnumpy()[0]
            #print probs

            action_index = action_select(act_dim, probs, epsilon)
            #print 'act', action_index, epsilon

            # scale down eplision
            if epsilon > final_epsilon:
                epsilon -= (initial_epsilon - final_epsilon) / args.anneal_epsilon_timesteps

            _, r_t, terminal = gamedata.act(action_index)

            action_index1 = np.zeros(act_dim)
            action_index1[action_index] = 1
            a_batch.append(action_index1.reshape((-1,act_dim)))


            r_t = np.array([r_t])
            ep_reward += r_t
            Done.append(terminal)

            past_rewards.append(r_t.reshape((-1, 1)))

            t += 1
            T += 1
            ep_t += 1

        if terminal:
            R_t = np.array([-0.2])
        else:
            value_out = module.get_outputs()[1]
            R_t = value_out.asnumpy()

        err = 0
        #print R_t
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + args.gamma * R_t

            #print 'value_loss', R_t, V[i]
            batch = mx.io.DataBatch(data=[mx.nd.array(s_batch[i]),
                                          mx.nd.array(R_t), mx.nd.array(a_batch[i])],
                                    label=None)
            #print mx.nd.array(s_batch[i]), mx.nd.array(R_t), mx.nd.array(a_batch[i])
            module.forward(batch, is_train=True)
            #print past_rewards[i], module.get_outputs()[3].asnumpy(), 'value_loss', module.get_outputs()[1].asnumpy(), 'log_policy', module.get_outputs()[0].asnumpy(), 'policy_out', module.get_outputs()[2].asnumpy()
            #print 'value_loss', module.get_outputs()[1].asnumpy(), 'value_out', module.get_outputs()[3].asnumpy(), R_t, past_rewards[i], 'policy_out', module.get_outputs()[2].asnumpy()
            total_loss = module.get_outputs()[2]
            #print 'total_loss', total_loss.asnumpy()
            module.backward()
            score += past_rewards[i]

        copyTargetQNetwork(module, Net)
        module.update()

        if terminal:
            print 'Thread, ', num, 'Eposide end! reward ', ep_reward, T
            ep_reward = 0
            terminal = False

        if args.save_every !=0 and epoch % args.save_every == 0:
            save_params(args.save_model_prefix, Net, epoch)

def train():

    global Net, lock
    Net = setup()
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
