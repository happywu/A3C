import mxnet as mx

def get_symbol_atari(act_dim, isQnet=False):
    data = mx.symbol.Variable('data')
    rewardInput = mx.sym.Variable('rewardInput')
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
    policy_log = mx.symbol.log(policy)
    policy_out = mx.symbol.BlockGrad(data=policy, name='policy_out')

    ## value network
    value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_loss = mx.symbol.MakeLoss((rewardInput - value_network) ** 2)
    value_out = mx.symbol.BlockGrad(data=value, name='value_out')

    # policy_log needs out_grad when backward
    return mx.symbol.Group([policy_log, value_loss, policy_out, value_out])

