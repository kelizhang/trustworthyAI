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

import itertools
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from castle.common import GradientBasedLearner, Tensor
from castle.common.independence_tests import hsic_test
from castle.common.class_factory import ClassFactory

from .utils import batch_loader, compute_jacobian, compute_entropy


class MLP(nn.Module):
    """
    Multi-layer perceptron

    """

    def __init__(self, input_dim, hidden_layers, hidden_units, output_dim,
                 bias=True, activation=None, device=None) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation
        self.device = device

        mlp = []
        for i in range(self.hidden_layers):
            input_size = self.hidden_units
            if i == 0:
                input_size = self.input_dim
            weight = nn.Linear(in_features=input_size,
                               out_features=self.hidden_units,
                               bias=self.bias,
                               dtype=torch.float,
                               device=self.device)
            mlp.append(weight)
            if self.activation is not None:
                mlp.append(self.activation)
        out_layer = nn.Linear(in_features=self.hidden_units,
                              out_features=self.output_dim,
                              bias=self.bias,
                              dtype=torch.float,
                              device=self.device)
        mlp.append(out_layer)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:

        out = self.mlp(x)

        return out


@ClassFactory.algorithms.register()
class PNL(GradientBasedLearner):
    """
    On the Identifiability of the Post-Nonlinear Causal Model

    References
    ----------
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2599.pdf

    Parameters
    ----------
    hidden_layers: int
        number of hidden layer of mlp
    hidden_units: int
        number of unit of per hidden layer
    batch_size: int
        size of training batch
    iterations: int
        training times on all samples
    lr: float
        learning rate
    alpha: float
        significance level
    bias: bool
        whether use bias
    activation: callable
        nonlinear activation function

    Examples
    --------
    >>> import castle
    >>> castle.set_backend('pytorch')
    >>> from castle.algorithms import PNL
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = PNL()
    # If you want to use GPU, you can call CORL().set_context() like the following code:
    >>> algo.set_context(device_target='GPU') # 'GPU', 'CPU' or 'Ascend'
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    def __init__(self, hidden_layers=1, hidden_units=10, batch_size=64,
                 iterations=100, lr=1e-4, alpha=0.01, bias=True,
                 activation=nn.LeakyReLU()):
        super(PNL, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr = lr
        self.alpha = alpha
        self.bias = bias
        self.activation = activation

    def learn(self, data, columns=None, **kwargs):

        data = Tensor(data, columns=columns).data
        n_nodes = data.shape[1]
        g = np.zeros((n_nodes, n_nodes))
        all_nodes_pair = itertools.permutations(range(n_nodes), 2)
        progress_bar = tqdm([*all_nodes_pair], disable=False)
        for i, j in progress_bar:
            x1 = torch.tensor(data[:, i], dtype=torch.float, device=self.device).unsqueeze(-1)
            x2 = torch.tensor(data[:, j], dtype=torch.float, device=self.device).unsqueeze(-1)

            # initialize model and parameters
            l1 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation,
                     device=self.device)
            l2 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation,
                     device=self.device)
            optimizer = torch.optim.SGD([{'params': l1.parameters()},
                                         {'params': l2.parameters()}],
                                        lr=self.lr)
            # nonlinear ICA
            e2 = self._nonlinear_ica(l1, l2, x1, x2, optimizer=optimizer)

            # kernel-based independent test
            ind = hsic_test(x1.cpu().detach().numpy(),
                            e2.cpu().detach().numpy(), alpha=self.alpha)
            if ind:  # x1->x2
                g[i, j] = 1

        self.causal_matrix = Tensor(g, index=columns, columns=columns)

    def _nonlinear_ica(self, f1, f2, x1, x2, optimizer):

        batch_generator = batch_loader(x1, x2, batch_size=self.batch_size)
        for _ in range(self.iterations):
            for x1_batch, x2_batch in batch_generator:
                optimizer.zero_grad()

                l2_jacob = torch.diag(compute_jacobian(f2, x2_batch).squeeze())
                e2 = f2(x2_batch) - f1(x1_batch)
                entropy = compute_entropy(e2)
                loss = entropy - torch.log(torch.abs(l2_jacob)).sum()

                loss.backward()
                optimizer.step()

        e2 = f2(x2) - f1(x1)

        return e2
