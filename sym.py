import mxnet as mx

def get_symbol_atari(act_dim, entropy_beta=0.01):
    data = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    ## policy network
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')
    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_out = mx.symbol.BlockGrad(data=value, name='value_out')
    # loss
    rewardInput = mx.symbol.Variable('rewardInput')
    actionInput = mx.symbol.Variable('actionInput')
    tdInput = mx.symbol.Variable('tdInput')
    # avoid NaN with clipping when pi becomes zero

    policy_log = mx.symbol.log(mx.symbol.clip(data=policy, a_min=1e-20, a_max=1.0))
    # add minus, because gradient ascend is used to optimize policy in the
    # paper, here we use gradient descent
    entropy =  - mx.symbol.sum(policy * policy_log, axis=1)

    policy_loss = - mx.symbol.sum(mx.symbol.sum(policy_log * actionInput,
                                                axis=1) * mx.symbol.sum(tdInput, axis=1) + entropy * entropy_beta)
    value_loss = mx.symbol.sum(mx.symbol.square(rewardInput - value))
    total_loss = mx.symbol.MakeLoss(policy_loss + (0.5 * value_loss))

    return mx.symbol.Group([policy_out, value_out, total_loss])

def get_symbol_atari_bn(act_dim, entropy_beta=0.01)
    data = mx.symbol.Variable('data')
    rewardInput = mx.symbol.Variable('rewardInput')
    actionInput = mx.symbol.Variable('actionInput')
    tdInput = mx.symbol.Variable('tdInput')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.BatchNorm(data=net, name='bn1')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu1', act_type="leaky")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.BatchNorm(data=net, name='bn2')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu2', act_type="leaky")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.BatchNorm(data=net, name='bn4')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu4', act_type='leaky')
    ## policy network
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1-1e-5)
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')
    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_out = mx.symbol.BlockGrad(data=value, name='value_out')
    # loss
    # avoid NaN with clipping when pi becomes zero

    policy_log = mx.symbol.log(data=policy, name='policy_log')
    # add minus, because gradient ascend is used to optimize policy in the
    # paper, here we use gradient descent
    entropy =  - mx.symbol.sum(policy * policy_log, axis=1)

    policy_loss = - mx.symbol.sum(mx.symbol.sum(policy_log * actionInput,
                                                axis=1) * mx.symbol.sum(tdInput, axis=1) + entropy * entropy_beta)
    value_loss = mx.symbol.sum(mx.symbol.square(rewardInput - value))

    policy_loss = mx.symbol.MakeLoss(policy_loss)
    value_loss = mx.symbol.MakeLoss(0.5 * value_loss)

    #total_loss = mx.symbol.MakeLoss(policy_loss + (0.5 * value_loss))
    total_loss = mx.symbol.BlockGrad(policy_loss + (0.5 * value_loss))

    return mx.symbol.Group([policy_out, value_out, total_loss, policy_loss, value_loss])


def get_symbol_test_bn(act_dim):
    data = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.BatchNorm(data=net, name='bn1')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu1', act_type="leaky")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.BatchNorm(data=net, name='bn2')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu2', act_type="leaky")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.BatchNorm(data=net, name='bn4')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu4', act_type='leaky')
    ## policy network
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    policy = mx.sym.clip(data=policy, a_min=1e-5, a_max=1-1e-5)
    policy_log = mx.symbol.log(data=policy, name='policy_log')
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')
    # Negative entropy
    neg_entropy = policy * policy_log
    neg_entropy = mx.sym.MakeLoss(data=neg_entropy, grad_scale=0.01, name='neg_entropy')

    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)

    return mx.sym.Group([policy_log, value, neg_entropy, policy_out])

def get_symbol_forward(act_dim):
    data = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    ## policy network
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')
    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_out = mx.symbol.BlockGrad(data=value, name='value_out')

    return mx.symbol.Group([policy_out, value_out])


def get_symbol_forward_bn(act_dim):
    data = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.BatchNorm(data=net, name='bn1')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu1', act_type="leaky")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.BatchNorm(data=net, name='bn2')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu2', act_type="leaky")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.BatchNorm(data=net, name='bn4')
    net = mx.symbol.LeakyReLU(data=net, name='leakyrelu4', act_type='leaky')
    ## policy network
    fc_policy = mx.symbol.FullyConnected(data=net, name='fc_policy', num_hidden=act_dim)
    policy = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')
    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_out = mx.symbol.BlockGrad(data=value, name='value_out')

    return mx.symbol.Group([policy_out, value_out])

def clipped_error(x):
    if mx.sym.abs(x)<1.0:
        return 0.5 * mx.sym.square(x)
    else:
        return mx.sym.abs(x) - 0.5
    
def get_dqn_symbol(act_dim, ispredict=False, clip_loss=False):
    data = mx.symbol.Variable('data')
    net = mx.symbol.Cast(data=data, dtype='float32')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    # Q Network
    Qvalue = mx.symbol.FullyConnected(data=net, name='qvalue', num_hidden=act_dim)
    Qvalue_out = mx.symbol.BlockGrad(data=Qvalue, name='qout')

    rewardInput = mx.symbol.Variable('rewardInput')
    actionInput = mx.symbol.Variable('actionInput')
    temp1 = mx.symbol.sum(Qvalue * actionInput, axis=1, keepdims=True, name='temp1')
    if clip_loss:
        loss = mx.sym.MakeLoss(clipped_error(rewardInput - temp1))
    else:
        loss = mx.symbol.MakeLoss(mx.symbol.square(rewardInput -
                                                   temp1))
    if (ispredict):
        # Target q network, only predict
        return Qvalue
    else :
        return mx.symbol.Group([loss, Qvalue_out])
