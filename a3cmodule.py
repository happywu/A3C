import logging
import warnings
import mxnet as mx
import numpy as np
from mxnet.module import Module
from mxnet import context as ctx
COUNT_MAX = 8
USE_AVERAGE = True

class A3CModule(Module):
    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None):
        # count how many times gradients be added
        self.add_counter = 0
        self.count_max = COUNT_MAX
        super(A3CModule, self).__init__(symbol=symbol, data_names=data_names,
                                        label_names=label_names, logger=logger, context=context,
                                        work_load_list=work_load_list, fixed_param_names=fixed_param_names, state_names=state_names)

    def clear_gradients(self):
        """clear gradient
        """
        self.add_counter = 0
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad -= grad

    def aver_gradients(self, n):
        ''' get average gradients
        '''
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad /= float(n)
    
    def add_gradients_from_module(self, from_module):
        """add gradients
        """
        self.add_counter += 1
        gradfrom = [[grad.copyto(grad.context) for grad in grads] for grads in
                    from_module._exec_group.grad_arrays]
        for gradsto, gradsfrom in zip(self._exec_group.grad_arrays,
                                      gradfrom):
            for gradto, gradfrom in zip(gradsto, gradsfrom):
                gradto += gradfrom

        if self.add_counter == self.count_max:
            if USE_AVERAGE:
                self.aver_gradients(self.add_counter)
                self.update()
                self.clear_gradients()
            self.add_counter = 0

    def copy_from_module(self, from_module):
        """copy from another module
        """
        arg_params, aux_params = from_module.get_params()
        self.init_params(initializer=None, arg_params=arg_params,
                         aux_params=aux_params, force_init=True)

    def clip_gradients(self, threshold):
        """clip gradients
        """
        for grads in self._exec_group.grad_arrays:
            for grad in grads:
                grad -= grad - \
                    mx.nd.clip(grad, -1.0 * threshold, 1.0 * threshold).copy()
