import mxnet as mx

def get_symbol_atari(act_dim, isQnet=False):
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
    policy_log = mx.symbol.log(mx.symbol.sum(policy * actionInput, axis=1))
    policy_loss = -policy_log * mx.symbol.sum(rewardInput - value)
    value_loss = mx.symbol.mean(mx.symbol.square(rewardInput - value))
    total_loss = mx.symbol.MakeLoss(policy_loss + (0.5 * value_loss))
    return mx.symbol.Group([policy_out, value_out]), total_loss

