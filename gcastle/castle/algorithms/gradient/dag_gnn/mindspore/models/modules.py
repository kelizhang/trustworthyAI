# coding = utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore import Parameter
from mindspore.scipy.linalg import inv
from mindspore.common.initializer import initializer, XavierUniform

from castle.algorithms.gradient.dag_gnn.mindspore.utils import functions as func


class MLP(nn.Cell):
    """
    Multilayer perceptron

    Parameters
    ----------
    input_dim: int, input layer dimension
    hidden_dim: int, hidden layer dimension
    output_dim: int, output layer dimension
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.w1 = nn.Dense(in_channels=self.input_dim,
                           out_channels=self.hidden_dim,
                           has_bias=True)
        self.w2 = nn.Dense(in_channels=self.hidden_dim,
                           out_channels=self.output_dim,
                           has_bias=True)

    def construct(self, x):
        """
        Calculate MLP result of input data
        """
        out_x = self.w1(x)
        relu = nn.ReLU()
        out_x = relu(out_x)
        out_x = self.w2(out_x)

        return out_x


class Encoder(nn.Cell):
    """
    Graph neural network model: encoder

    Parameters
    ----------
    adj_A: mindspore.Tensor
        weighted adjacency matrix with shape (n_nodes, n_nodes)
    input_dim: int
        encoder input dimension
    hidden_dim: int
        encoder hidden dimension
    output_dim: int
        encoder output dimension
    encoder_type: str
        choose an encoder, 'mlp' or 'sem'
    """
    def __init__(self, adj_A, input_dim, hidden_dim, output_dim, encoder_type):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        if self.encoder_type == 'mlp':
            self.encoder_mlp = MLP(input_dim, hidden_dim, output_dim)
            self._init_encoder_weights()

        # Initialize encoder hyper-parameters
        self.adj_A = Parameter(adj_A, requires_grad=True)
        self.wa = Parameter(msnp.zeros(output_dim), requires_grad=True)

    def _init_encoder_weights(self):
        """
        Initialize encoder weights
        """
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight = initializer(XavierUniform(), shape=m.weight.shape)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight = m.weight.fill(1)
                m.bias = msnp.zeros(m.bias.shape)

    def construct(self, x):
        """
        Encoder forward network
        """
        # to amplify the value of A and accelerate convergence.
        adj_A1 = ops.sinh(3. * self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = msnp.eye(adj_A1.shape[0]) - adj_A1.T
        if self.encoder_type == 'mlp':
            mlp_out = self.encoder_mlp(x)
            logits = ops.matmul(adj_Aforz, mlp_out + self.wa) - self.wa
        else:
            adj_A_inv = inv(adj_Aforz)
            meanF = ops.matmul(adj_A_inv, ops.mean(ops.matmul(adj_Aforz, x), 0))
            logits = ops.matmul(adj_Aforz, x - meanF)

        return logits, adj_A1


class Decoder(nn.Cell):
    """
    Graph neural network model: decoder

    Parameters
    ----------
    input_dim: int
        decoder input dimension
    hidden_dim: int
        decoder hidden dimension
    output_dim: int
        decoder output dimension
    decoder_type: str
        choose a decoder, 'mlp' or 'sem'
    """
    def __init__(self, input_dim, hidden_dim, output_dim, decoder_type):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.decoder_type = decoder_type
        if self.decoder_type == 'mlp':
            self.decoder_mlp = MLP(input_dim, hidden_dim, output_dim)
            self._init_decoder_weights()

    def _init_decoder_weights(self):
        """
        Initialize decoder weights
        """
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight = initializer(XavierUniform(), shape=m.weight.shape)
                m.bias = m.bias.fill(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight = m.wight.fill(1)
                m.bias = msnp.zeros(m.bias.shape)

    def construct(self, logits, origin_a, wa):
        """
        Decoder forward network
        """
        # adj_A_new = (I-A^T)^(-1)
        adj_A_new = inv(msnp.eye(origin_a.shape[0]) - origin_a.T)
        mat_z = ops.matmul(adj_A_new, logits + wa) - wa
        if self.decoder_type == 'mlp':
            out = self.decoder_mlp(mat_z)
        else:
            out = mat_z

        return out


class DagGnnLoss(nn.Cell):
    """
    Loss for network train
    """
    def __init__(self, c_a, lambda_a, graph_threshold, n_nodes, tau_a,
                 use_a_connect_loss, use_a_positiver_loss):
        super(DagGnnLoss, self).__init__()
        self.c_a = c_a
        self.lambda_a = lambda_a
        self.graph_threshold = graph_threshold
        self.n_nodes = n_nodes
        self.tau_a = tau_a
        self.use_a_connect_loss = use_a_connect_loss
        self.use_a_positiver_loss = use_a_positiver_loss

        if self.use_a_connect_loss:
            self.z_gap = Parameter(ms.Tensor(0.1), requires_grad=True)
        if self.use_a_positiver_loss:
            self.z_positive = Parameter(ops.ones_like(self.adj_A),
                                        requires_grad=True)

        self.loss_nll = Parameter(ms.Tensor(0.0), name='loss_nll',
                                  requires_grad=False)
        self.loss_kl = Parameter(ms.Tensor(0.0), name='loss_kl',
                                 requires_grad=False)
        self.origin_a = Parameter(ms.Tensor(
            msnp.zeros((self.n_nodes, self.n_nodes))),
            name='origin_a', requires_grad=False)

    def construct(self, x, x_pred, logits, origin_a):
        """
        Loss forward network

        Parameters
        ----------
        x: mindspore.Tensor
            origin samples data
        x_pred: mindspore.Tensor
            output from decoder network
        logits: mindspore.Tensor
            output from encoder network
        origin_a: mindspore.Tensor
            weighted adjacency matrix with shape (n_nodes, n_nodes)

        Returns
        -------
        loss: loss value
        """
        # reconstruction accuracy loss
        loss_nll = func.nll_gaussian(x_pred, x)

        # KL loss
        loss_kl = func.kl_gaussian_sem(logits)

        # ELBO loss
        loss = loss_kl + loss_nll

        # add A loss
        one_adj_a = origin_a
        sparse_loss = self.tau_a * msnp.sum(msnp.abs(one_adj_a))

        # other loss term
        if self.use_a_connect_loss:
            connect_gap = func.a_connect_loss(one_adj_a, self.graph_threshold, self.z_gap)
            loss += self.lambda_a * connect_gap + 0.5 * self.c_a * connect_gap * connect_gap

        if self.use_a_positiver_loss:
            positive_gap = func.a_positive_loss(one_adj_a, self.z_positive)
            loss += .1 * (self.lambda_a * positive_gap
                          + 0.5 * self.c_a * positive_gap * positive_gap)

        # compute h(A)
        h_A = func._h_A(origin_a, self.n_nodes)
        # augmented Lagrangian
        loss += (self.lambda_a * h_A
                 + 0.5 * self.c_a * h_A * h_A
                 + 100. * msnp.trace(origin_a * origin_a)
                 + sparse_loss)

        self.origin_a = origin_a
        self.loss_nll = loss_nll
        self.loss_kl = loss_kl

        return loss


class NetWithLossCell(nn.Cell):
    """
    Packaging network and loss

    Parameters
    ----------
    encoder: Encoder network instantiate object
    decoder: Decoder network instantiate object
    loss_fn: DagGnnLoss instantiate object
    """
    def __init__(self, encoder, decoder, loss_fn):
        super(NetWithLossCell, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def construct(self, x):
        logits, origin_a = self.encoder(x)
        x_pred = self.decoder(logits, origin_a, self.encoder.wa)
        loss = self.loss_fn(x, x_pred, logits, origin_a)

        return loss
