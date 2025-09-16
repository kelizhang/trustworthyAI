# coding=utf-8
# Huawei Technologies Co., Ltd.
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Xun Zheng (https://github.com/xunzheng/notears)
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

import math
import numpy as np

import mindspore
from mindspore.common.initializer import initializer, Zero, HeNormal, Uniform, _calculate_fan_in_and_fan_out
from mindspore import ops, Parameter, numpy, nn, ParameterTuple
from mindspore.ops import functional as F

from .utils.locally_connected import LocallyConnected


def init_bias(weight):
    """
        Initialize the bias based on the weight.
    Parameters
    ----------
    weight: initializer
        Weight parameter
    Returns
    -------
    :initializer
        bias parameter
    """
    fan_in, _ = _calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    bias = initializer(Uniform(bound), [weight.shape[0]])
    return bias


class MLPModel(nn.Cell):
    def __init__(self, dims, bias=True):
        """
        Multilayer perceptron.

        Parameters
        ----------
        dims: tuple
            Network shape parameters
        bias:
            Indicates whether to use weight deviation.
        """
        super(MLPModel, self).__init__()
        if len(dims) < 2:
            raise ValueError(f"The size of dims at least greater equal to 2, contains one "
                             f"one hidden layer and one output_layer")
        if dims[-1] != 1:
            raise ValueError(f"The dimension of output layer must be 1, but got {dims[-1]}.")
        d = dims[0]
        self.dims = dims

        weight_pos = initializer(HeNormal(math.sqrt(5)), [d * dims[1], d])
        bias_pos = init_bias(weight_pos)

        weight_neg = initializer(HeNormal(math.sqrt(5)), [d * dims[1], d])
        bias_neg = init_bias(weight_neg)

        self.fc1_pos = nn.Dense(d, d * dims[1], has_bias=bias, weight_init=weight_pos, bias_init=bias_pos)
        self.fc1_neg = nn.Dense(d, d * dims[1], has_bias=bias, weight_init=weight_neg, bias_init=bias_neg)

        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()

        # fc2: local linear layers
        layers = []
        for layer in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[layer + 1], dims[layer + 2], bias=bias))
        self.fc2 = nn.CellList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for _ in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def construct(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = ops.Sigmoid()(x.astype(mindspore.float32)) # [n, d, m1]  
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(axis=2)  # [n, d]
        return x

    def h_func(self):
        """
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG.

        Returns
        -------

        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = ops.ReduceSum()(fc1_weight * fc1_weight, 1).T  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        init_e = ops.eye(d, d, mindspore.float32)
        M = init_e + A / d  # (Yu et al. 2019)
        E = numpy.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        return h

    def l2_reg(self):
        """
        Take 2-norm-squared of all parameters.

        Returns
        -------

        """
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += ops.ReduceSum()(fc1_weight ** 2)
        for fc in self.fc2:
            reg += ops.ReduceSum()(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """
        Take l1 norm of fc1 weight.

        Returns
        -------

        """
        reg = ops.ReduceSum()(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """
        Get W from fc1 weights, take 2-norm over m1 dim.

        Returns
        -------

        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = ops.ReduceSum()(fc1_weight * fc1_weight, 1).T  # [i, j] 
        W = ops.Sqrt()(A)  # [i, j]
        W = W.asnumpy()  # [i, j]
        return W


class SobolevModel(nn.Cell):
    """
    Sobolev network.

    Parameters
    ----------
    d: int
        Num variables.
    k: int
        Num expansion of each variable.
    bias:
        Indicates whether to use weight deviation.
    device: option, default: None
        torch.device('cpu') or torch.device('cuda')
    """
    def __init__(self, d, k, bias=False):
        super().__init__()
        self.d, self.k = d, k
        self.l2_reg_store = Parameter(mindspore.Tensor(np.zeros((), dtype=np.float32)), name="l2_reg_store")
        self.fc1_pos = nn.Dense(d * k, d, has_bias=bias)
        self.fc1_neg = nn.Dense(d * k, d, has_bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        self.fc1_pos.weight = initializer(Zero(), self.fc1_pos.weight.shape)
        self.fc1_neg.weight = initializer(Zero(), self.fc1_neg.weight.shape)

    def _bounds(self):
        # weight shape [j, ik]
        bounds = []
        for j in range(self.d):
            for i in range(self.d):
                for _ in range(self.k):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def sobolev_basis(self, x):  # [n, d] -> [n, dk]
        seq = []
        for kk in range(self.k):
            mu = 2.0 / (2 * kk + 1) / math.pi  # sobolev basis
            psi = mu * ops.Sin()(x / mu)
            seq.append(psi)  # [n, d] * k
        bases = ops.Stack(axis=2)(seq)  # [n, d, k]
        bases = bases.view(-1, self.d * self.k)  # [n, dk]
        return bases

    def construct(self, x):  # [n, d] -> [n, d]
        bases = self.sobolev_basis(x)  # [n, dk]
        x = self.fc1_pos(bases) - self.fc1_neg(bases)  # [n, d]
        self.l2_reg_store = ops.ReduceSum()(x ** 2) / x.shape[0]
        return x 

    def h_func(self):
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = ops.ReduceSum()(fc1_weight * fc1_weight, 2).T  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        init_e = ops.eye(self.d, self.d, mindspore.float32)
        M = init_e + A / self.d  # (Yu et al. 2019)
        E = numpy.matrix_power(M, self.d - 1)
        h = (E.T * M).sum() - self.d
        return h

    def l2_reg(self):
        reg = self.l2_reg_store
        return reg

    def fc1_l1_reg(self):
        reg = ops.ReduceSum()(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    def fc1_to_adj(self) -> np.ndarray:
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j, ik]
        fc1_weight = fc1_weight.view(self.d, self.d, self.k)  # [j, i, k]
        A = ops.ReduceSum()(fc1_weight * fc1_weight, 2).T  # [i, j] 
        W = ops.Sqrt()(A)  # [i, j]
        W = W.asnumpy()  # [i, j]
        return W


class L1Loss(nn.Cell):
    def __init__(self):
        """
        least squares loss function

        """
        super(L1Loss, self).__init__()

    def construct(self, base, target):
        n = target.shape[0]
        loss = 0.5 / n * ops.ReduceSum()((base - target) ** 2)
        return loss


class NotearsWithLossCell(nn.WithLossCell):
    def __init__(self, backbone, loss_fn):
        """
        Perform the network and return loss.

        Parameters
        ----------
        backbone: nn.Module
            network model
        loss_fn:
            loss function
        """
        super(NotearsWithLossCell, self).__init__(backbone, loss_fn)

    def construct(self, data, label, p_dict):
        """
            Only the operators placed in the construct function have the influence with the gradient.
        Parameters
        ----------
        data
        label
        p_dict

        Returns
        -------

        """
        rho = p_dict["rho"]
        alpha = p_dict["alpha"]
        lambda2 = p_dict["lambda2"]
        lambda1 = p_dict["lambda1"]
        out = self._backbone(data)
        loss = self._loss_fn(out, label)
        h_val = self._backbone.h_func()
        fc1_l2_reg = self._backbone.l2_reg()
        fc1_l1_reg = self._backbone.fc1_l1_reg()
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        l2_reg = 0.5 * lambda2 * fc1_l2_reg
        l1_reg = lambda1 * fc1_l1_reg
        primal_obj = loss + penalty + l2_reg + l1_reg
        return primal_obj


class GradWrap(nn.Cell):
    def __init__(self, network):
        """
        Find the input gradient of the function

        Parameters
        ----------
        network: nn.Cell
            network model

        Returns
        -------
        :tuple
            Gradient of loss versus weight.
        """
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        # Wrap weight in Tuple form
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))
        self.grad = ops.GradOperation(get_by_list=True)
        self.grad_reducer = F.identity

    def construct(self, x, label, p_dict):
        weights = self.weights
        # The return value is gradient.
        grads = self.grad(self.network, weights)(x, label, p_dict)
        grads = self.grad_reducer(grads)
        return grads
