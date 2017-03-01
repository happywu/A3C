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
parser.add_argument('--num-threads', type=int, default=16)

args = parser.parse_args()

def sample_policy_action(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs.asnumpy()
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index

def actor_learner_thread(num, module, dataiter):
    act_dim = dataiter.act_dim
    time.sleep(5*num)
    dataiter.reset()

    for _ in range(100):
        data = dataiter.data()

        module.forward(mx.io.DataBatch(data=data, label=None), is_train=False)
        action_index = np.random.choice(act_dim)
        dataiter.act([action_index])

def setup():

    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    dataiters = [rl_data.GymDataIter(args.game, args.input_length, web_viz=False) for _ in range(args.num_threads)]
    act_dim = dataiters[0].act_dim
    net = sym.get_symbol_atari(act_dim)
    module = mx.mod.Module(net, data_names=[d[0] for d in
                                            dataiters[0].provide_data],
                           label_names=(['policy_label', 'value_label']), context=devs)

    module.bind(data_shapes=dataiters[0].provide_data,
                label_shapes=[('policy_label', (args.batch_size, )),
                              ('value_label', (args.batch_size, 1))],
                grad_req='add')

    return module, dataiters

def log_config(log_dir=None, log_file=None, prefix=None, rank=0):
    reload(logging)
    head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    if log_dir:
        logging.basicConfig(level=logging.DEBUG, format=head)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not log_file:
            log_file = (prefix if prefix else '') + datetime.now().strftime('_%Y_%m_%d-%H_%M.log')
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

def train(module, dataiters):

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

    module.init_params()
    module.init_optimizer(kvstore=kv, optimizer='adam',
                          optimizer_params={'learning_rate': args.lr, 'wd': args.wd, 'epsilon': 1e-3})
    # logging
    np.set_printoptions(precision=3, suppress=True)

    actor_learner_threads = [threading.Thread(target=actor_learner_thread, args=(thread_id, module, dataiters[thread_id])) for thread_id in range(args.num_threads)]

    for t in actor_learner_threads:
        t.start()

    for t in actor_learner_threads:
        t.join()

if __name__ == '__main__':
    module, dataiters = setup()
    train(module, dataiters)
