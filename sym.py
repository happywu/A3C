import mxnet as mx

def get_symbol_atari(act_dim, isQnet=False):
    data = mx.symbol.Variable('data')
    actionInput = mx.sym.Variable('actionInput')
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
    policy_network = mx.symbol.SoftmaxActivation(data=fc_policy, name='policy')
    ## value network
    fc_value = mx.symbol.FullyConnected(data=net, name='fc_value', num_hidden=1)
    value_network = mx.symbol.LinearRegressionOutput(data=fc_value, name='value')

    policy_temp = policy_network * actionInput
    policy_temp = mx.symbol.sum(policy_temp, axis=1)

    policy_loss = mx.symbol.MakeLoss(mx.symbol.log(policy_temp) * (rewardInput - value_network))
    value_loss = mx.symbol.MakeLoss((rewardInput - value_network) ** 2)

    #if isQnet:
    return mx.symbol.Group(value_network, policy_network, value_loss, policy_loss)
    #else:
    #    return mx.symbol.Group(policy_network, policy_loss)
    #return mx.symbol.Group([policy_network, value_network])
    #return mx.symbol.Group([policy_loss, value_loss])

