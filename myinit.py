from mxnet.initializer import register
from mxnet.initializer import Initializer
from mxnet import random
import numpy as np

# adapt from [universe-starter-agent](https://github.com/openai/universe-starter-agent)
@register
class normalized_columns_initializer(Initializer):
    def __init__(self, std=1.0):
        super(normalized_columns_initializer, self).__init__()
        self.std = std

    def _init_weight(self, _, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        out = np.random.randn(int(fan_out), int(fan_in)).astype(np.float32)
        out *= self.std / np.sqrt(np.square(out).sum(axis=1, keepdims=True))
        arr[:] = out

@register
class my_conv_initializer(Initializer):
    def __init__(self, std=1.0):
        super(my_conv_initializer, self).__init__()


    def _init_weight(self, _, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        random.uniform(-w_bound, w_bound, out=arr)

