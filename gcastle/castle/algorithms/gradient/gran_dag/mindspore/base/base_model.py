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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.probability.distribution import Normal
from mindspore.common.initializer import initializer, XavierUniform


class GranLoss(LossBase):
    """
    Loss class for network train

    Parameters
    ----------
    input_dim: int
        number of input layer (number of variables), must be int
    mu: float
        initialization of penalty coefficient in the optimization of
        augmented Lagrangian
    lamb: float
        initialization of Lagrangian coefficient in the optimization of
        augmented Lagrangian
    norm_prod: str
        use norm of product of paths, 'none' or 'paths'
        'paths': use norm, 'none': with no norm
    square_prod: bool
        use squared product of paths
    run_mode: str, default 'graph'
        run mode of code: 'graph' or 'pynative'
        'graph' is static graph mode, 'pynative' is dynamic graph mode
    """
    def __init__(self, input_dim, mu, lamb, norm_prod, square_prod, run_mode):
        super(GranLoss, self).__init__()
        """Initialize GranLoss."""
        self.input_dim = input_dim
        self.mu = mu
        self.lamb = lamb
        self.norm_prod = norm_prod
        self.square_prod = square_prod
        self.run_mode = run_mode

        # initialize current adjacency matrix
        self.adjacency = msnp.ones(self.input_dim) - msnp.eye(self.input_dim)
        if self.run_mode == 'graph':
            self.k = ms.Parameter(ms.Tensor(1), requires_grad=False)

    def get_w_adj(self, weights):
        """Get weighted adjacency matrix"""
        return self.compute_A_phi(weights, norm=self.norm_prod, square=self.square_prod)

    def compute_A_phi(self, weights, norm="none", square=False):
        """
        compute matrix A consisting of products of NN weights

        Parameters
        ----------
        weights: list of net weights
        norm: str, default 'none'
            use norm of product of paths, 'none' or 'paths'
            'paths': use norm, 'none': with no norm
        square: bool, default False
            use squared product of paths
        """

        prod = msnp.eye(self.input_dim)
        if norm == "paths":
            prod_norm = msnp.eye(self.input_dim)
            for i, w in enumerate(weights):
                if square:
                    w = w ** 2
                else:
                    w = ops.absolute(w)
                if i == 0:
                    tmp_adj = ops.expand_dims(self.adjacency.transpose(), 1)
                    ein_one = ops.mul(w, tmp_adj)
                    prod = ops.matmul(ein_one, prod)
                    tmp_adj = 1. - msnp.eye(self.input_dim)
                    tmp_adj = ops.expand_dims(tmp_adj.transpose(), 1)
                    ein_two = ops.mul(ops.ones_like(w), tmp_adj)
                    prod_norm = ops.matmul(ein_two, prod_norm)
                else:
                    prod = ops.matmul(w, prod)
                    prod_norm = ops.matmul(ops.ones_like(w), prod_norm)

            # sum over density parameter axis
            prod = msnp.sum(prod, 1)
            prod_norm = msnp.sum(prod_norm, 1)
            # avoid / 0 on diagonal
            denominator = prod_norm + msnp.eye(self.input_dim)
            return (prod / denominator).transpose()
        elif norm == "none":
            for i, w in enumerate(weights):
                if square:
                    w = w ** 2
                else:
                    w = ops.absolute(w)
                if i == 0:
                    tmp_adj = ops.expand_dims(self.adjacency.transpose(), 1)
                    ein_one = ops.mul(w, tmp_adj)
                    prod = ops.matmul(ein_one, prod)
                else:
                    prod = ops.matmul(w, prod)
            # sum over density parameter axis
            prod = msnp.sum(prod, 1)
            return prod.transpose()
        else:
            raise ValueError("Param 'norm_prod' should be in {'paths', 'none'}.")

    def compute_constraint(self, w_adj):
        """
        compute constraint value of weighted adjacency matrix
        constraint value: matrix exponent of w_adj minus num_vars

        Parameters
        ----------
        w_adj: mindspore.Tensor
            weighted adjacency matrix

        Returns
        -------
        h: constraint value
        """

        if self.run_mode == 'graph':
            expm_input = self.get_expm_graph(w_adj)
        else:
            expm_input = self.get_expm_pynative(w_adj)
        h = msnp.trace(expm_input) - self.input_dim

        return h

    def get_expm_graph(self, A):
        """
        Compute the matrix exponential of a square matrix.

        Parameters
        ----------
        A: mindspore.Tensor
            a square matrix

        Returns
        -------
        expm_val: mindspore.Tensor
            matrix exponent value
        """
        expm_val = msnp.zeros(A.shape)
        eye_mat = msnp.eye(A.shape[0])
        self.k = 1

        if msnp.norm(eye_mat, 1) <= 0:
            return expm_val
        while True:
            expm_val += eye_mat
            eye_mat = msnp.matmul(A, eye_mat) / self.k
            self.k += 1
            if msnp.norm(eye_mat, 1) <= 0:
                break

        return expm_val

    @staticmethod
    def get_expm_pynative(A):
        """
        Compute the matrix exponential of a square matrix.

        Parameters
        ----------
        A: mindspore.Tensor
            a square matrix

        Returns
        -------
        expm_val: mindspore.Tensor
            matrix exponent value
        """
        expm_val = msnp.zeros(A.shape)
        eye_mat = msnp.eye(A.shape[0])
        k = 1

        if msnp.norm(eye_mat, 1) <= 0:
            return expm_val
        while True:
            expm_val += eye_mat
            eye_mat = msnp.matmul(A, eye_mat) / k
            k += 1
            if msnp.norm(eye_mat, 1) <= 0:
                break

        return expm_val

    def construct(self, inputs, *args):
        """
        Applying augmented Lagrangian to solve the continuous constrained problem.

        Parameters
        ----------
        inputs: tuple, (log_likelihood, weights)
            the value output from network

        Returns
        -------
        aug_lagrangian: augmented lagrangian value
        """
        log_likelihood, weights = inputs[0], inputs[1]
        loss_value = -ops.mean(log_likelihood)

        # calculate constraint
        w_adj = self.get_w_adj(weights)
        h = self.compute_constraint(w_adj)

        # compute augmented Lagrangian
        aug_lagrangian = loss_value + 0.5 * self.mu * h ** 2 + self.lamb * h

        return aug_lagrangian


class BaseModel(nn.Cell):
    """Base class of LearnableModel, disable instantiation."""

    def __init__(self, input_dim, hidden_num, hidden_dim, output_dim,
                 nonlinear="leaky-relu"):

        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinear = nonlinear

        self.normal = Normal()
        self.extra_params = []

        # initialize current adjacency matrix
        self.adjacency = msnp.ones(self.input_dim) - msnp.eye(self.input_dim)

        # Generate layer_list
        layer_list = [self.hidden_dim] * self.hidden_num
        layer_list.insert(0, self.input_dim)
        layer_list.append(self.output_dim)

        # Instantiate the parameters of each layer in the model of each variable
        tmp_weights = list()
        tmp_biases = list()
        for i, item in enumerate(layer_list[:-1]):
            in_dim = item
            out_dim = layer_list[i + 1]
            tmp_weights.append(ms.Parameter(msnp.zeros(
                (self.input_dim, out_dim, in_dim)),
                requires_grad=True, name='w'+str(i)))
            tmp_biases.append(ms.Parameter(msnp.zeros(
                (self.input_dim, out_dim)),
                requires_grad=True, name='b'+str(i)))

        self.weights = ms.ParameterTuple(tmp_weights)
        self.biases = ms.ParameterTuple(tmp_biases)

        # reset initialization parameters
        self.reset_params()

    def compute_log_likelihood(self, x):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution
            only if the DAG constraint on the mask is satisfied,
            otherwise the joint does not integrate to one.

        Parameters
        ----------
        x: mindspore.Tensor
            (batch_size, input_dim)

        Returns
        -------
        (batch_size, input_dim) log-likelihoods
        """
        weights, biases, extra_params = self.get_weight_bias_parameters(
            mode=('w', 'b', 'x'))
        density_params = self.forward_given_params(x, weights, biases)

        log_probs = []
        for i in range(self.input_dim):
            x_d = x[:, i]
            if len(extra_params) != 0:
                lp = self.get_distribution(
                    x_d, density_params[i].view(density_params[i].shape[0]),
                    extra_params[i])
            else:
                density_param = ops.Unstack(axis=1)(density_params[i])
                lp = self.get_distribution(
                    x_d, density_param[0], density_param[1])
            log_probs.append(ops.expand_dims(lp, 1))

        return ops.Concat(axis=1)(log_probs)

    def forward_given_params(self, x, weights, biases):
        """
        Compute output value of the fully connected NNs

        Parameters
        ----------
        x: batch_size x num_vars
        weights: List
            ith list contains weights for ith MLP
        biases: List
            ith list contains biases for ith MLP

        Returns
        -------
        out: batch_size x num_vars * num_params
            the parameters of each variable conditional
        """
        for k in range(self.hidden_num + 1):
            # apply affine operator
            if k == 0:
                # first part
                adj = ops.expand_dims(self.adjacency.transpose(), 1)
                einsum_one = ops.mul(weights[k], adj)
                # second part
                x = ops.expand_dims(ops.expand_dims(x, 2), 1)
                x = ops.matmul(einsum_one, x).squeeze(3) + biases[k]
            else:
                x = ops.matmul(weights[k],
                               ops.expand_dims(x, 3)).squeeze(3) + biases[k]

            # apply non-linearity element-wise
            if k != self.hidden_num:
                if self.nonlinear == "leaky-relu":
                    x = nn.LeakyReLU(alpha=0.01)(x)
                else:
                    x = nn.Sigmoid()(x)

        return ops.Unstack(axis=1)(x)

    def reset_params(self):
        """reset initialize parameter of network"""
        for node in range(self.input_dim):
            for i, w in enumerate(self.weights):
                w = w[node]
                tmp_w = initializer(XavierUniform(), shape=w.shape)
                self.weights[i][node] = tmp_w
            for i, b in enumerate(self.biases):
                b = b[node]
                tmp_b = msnp.zeros((b.shape[0]))
                self.biases[i][node] = tmp_b

    def get_weight_bias_parameters(self, mode=('w', 'b', 'x')):
        """
        Will get only parameters with requires_grad == True

        Parameters
        ----------
        mode: tuple
            w=weights, b=biases, x=extra_params (order is irrelevant)

        Returns
        -------
        params: list, corresponding list of list of parameters
        """

        params = list()

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b' in mode:
            biases = []
            for j, b in enumerate(self.biases):
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                extra_params.append(ep)
            params.append(extra_params)

        return params

    def set_parameters(self, params, mode="wbx"):
        """
        Will set only parameters with requires_grad == True

        Parameters
        ----------
        params: list of Parameters tuple to set,
            the order should be coherent with `get_parameters`
        mode: str
            w=weights, b=biases, x=extra_params (order is irrelevant)
        """

        k = 0
        if 'w' in mode:
            weights = params[k]
            for node in range(self.input_dim):
                for i in range(len(self.weights)):
                    self.weights[i][node] = weights[i][node]
            k += 1

        if 'b' in mode:
            biases = params[k]
            for node in range(self.input_dim):
                for i in range(len(self.biases)):
                    self.biases[i][node] = biases[i][node]
            k += 1

        if 'x' in mode and len(self.extra_params) > 0:
            extra_params = params[k]
            for i, ep in enumerate(self.extra_params):
                if ep.requires_grad:
                    self.extra_params[i][0] = extra_params[i][0]

    def construct(self, x, log_flag=False):
        """
        forward network, used to compute log_likelihood value
        """
        # calculate log likelihood
        log_likelihood = self.compute_log_likelihood(x)
        if log_flag:
            return log_likelihood
        weights = self.get_weight_bias_parameters(mode=('w', ))[0]

        return log_likelihood, weights


class LearnableModel(BaseModel):
    """Class for other learnable Models, disable instantiation."""

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu"):
        super(LearnableModel, self).__init__(input_dim=input_dim,
                                             hidden_num=hidden_num,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             nonlinear=nonlinear)


class NonlinearGauss(LearnableModel):
    """
    Class of learnable models based on NonlinearGauss

    Parameters
    ----------
    input_dim: int
        number of features
    hidden_num: int
        number of hidden layers
    hidden_dim: int
        number of dimension per hidden layer
    output_dim: int
    nonlinear: str, default 'leaky-relu'
        name of Nonlinear activation function
    """

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu"):
        super(NonlinearGauss, self).__init__(input_dim=input_dim,
                                             hidden_num=hidden_num,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             nonlinear=nonlinear)

    def get_distribution(self, x_d, dp_mean, dp_std):
        return self.normal.log_prob(x_d, dp_mean, ops.exp(dp_std))


class NonlinearGaussANM(LearnableModel):
    """
    Class of learnable models based on NonlinearGaussANM

    Parameters
    ----------
    input_dim: int
        number of features
    hidden_num: int
        number of hidden layers
    hidden_dim: int
        number of dimension per hidden layer
    output_dim: int
    nonlinear: str, default 'leaky-relu'
        name of Nonlinear activation function
    """

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu"):
        super(NonlinearGaussANM, self).__init__(input_dim=input_dim,
                                                hidden_num=hidden_num,
                                                hidden_dim=hidden_dim,
                                                output_dim=output_dim,
                                                nonlinear=nonlinear)

        # extra parameters are log_std
        extra_params = np.ones(self.input_dim)
        np.random.shuffle(extra_params)
        extra_params_list = list()
        for i, extra_param in enumerate(extra_params):
            extra_params_list.append(ms.Parameter(
                ms.Tensor(np.log(extra_param).reshape(1), dtype=ms.float32),
                requires_grad=True, name='e' + str(i)))

        # each element in the list represents a variable,
        # the size of the element is the number of extra_params per var
        self.extra_params = ms.ParameterTuple(extra_params_list)

    def get_distribution(self, x_d, dp_mean, dp_std):
        return self.normal.log_prob(x_d, dp_mean, ops.exp(dp_std))
