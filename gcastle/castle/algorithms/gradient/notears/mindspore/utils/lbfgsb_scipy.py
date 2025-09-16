# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mindspore
import scipy.optimize as sopt
from mindspore.nn import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore import ops, numpy, Tensor


class LBFGSBScipy(Optimizer):
    """
    Wrap L-BFGS-B algorithm, using scipy routines.
    
    Courtesy: Arthur Mensch's gist
    https://gist.github.com/arthurmensch/c55ac413868550f89225a0b9212aa4cd
    """

    @opt_init_args_register
    def __init__(self, network, params, loss_scale=1.0, learning_rate=1e-3):
        super(LBFGSBScipy, self).__init__(learning_rate, params, weight_decay=0.0, loss_scale=loss_scale)
        self.network = network
        self.assign = ops.Assign()
        self._numel = sum([p.size for p in self.parameters])

    @staticmethod
    def _gather_flat_grad(grads):
        views = list()
        for g in grads:
            view = g.view(-1)
            views.append(view)
        flat_grad = numpy.concatenate(views, axis=0)
        return flat_grad

    def _gather_flat_bounds(self):
        bounds = []
        for p in self.parameters:
            if hasattr(p, 'bounds'):
                b = p.bounds
            else:
                b = [(None, None)] * p.size
            bounds += b
        return bounds

    def _gather_flat_params(self):
        views = []
        for p in self.parameters:
            view = p.data.view(-1)
            views.append(view)
        return ops.Concat(0)(views)

    def _distribute_flat_params(self, params):
        offset = 0
        for i, p in enumerate(self.parameters):
            numel = p.size
            # view as to avoid deprecated pointwise semantics
            self.parameters[i].set_data(data=params[offset:offset + numel].reshape(p.data.shape))
            offset += numel
        assert offset == self._numel

    def step(self, closure):
        """
        Performs a single optimization step.

        Parameters
        ----------
        closure: callable
            A closure that reevaluates the model and returns the loss.
        """
        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = Tensor(flat_params, dtype=mindspore.float32)
            self._distribute_flat_params(flat_params)
            loss, grads = closure()
            loss = loss.asnumpy()
            flat_grad = self._gather_flat_grad(grads).asnumpy()
            return loss, flat_grad.astype('float64')

        initial_params = self._gather_flat_params()
        initial_params = initial_params.asnumpy()
        bounds = self._gather_flat_bounds()

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds)
        final_params = Tensor(sol.x.astype('float32'))
        self._distribute_flat_params(final_params)
        return sol.fun
