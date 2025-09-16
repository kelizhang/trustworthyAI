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

import gc
import itertools
import numpy as np
from tqdm import tqdm
import mindspore as ms
from mindspore import nn, ops
import mindspore.dataset as ds
import mindspore.numpy as msnp
from mindspore.nn import LossBase
from mindspore.nn.probability.distribution import Normal
from castle.common import GradientBasedLearner, Tensor
from castle.common.independence_tests import hsic_test
from castle.common.class_factory import ClassFactory


class PNLLoss(LossBase):
    """Custom loss function"""

    def __init__(self, f1, f2, reduction='none'):
        super(PNLLoss, self).__init__(reduction)
        self._f1 = f1
        self._f2 = f2
        self.normal = Normal()
        self.grad_op = ops.GradOperation()
        self.abs = ops.Abs()
        self.log = ops.Log()

    def compute_entropy(self, x):
        """Computation information entropy of x"""
        mean = msnp.mean(x)
        std = msnp.std(x)
        entropy = self.normal.entropy(mean, std)

        return entropy

    def compute_jacobian(self, inputs):
        """
        Function that computes the Jacobian of a given function.
        """
        gradient_function = self.grad_op(self._f2)
        jacobian = gradient_function(inputs).squeeze()

        return jacobian

    def construct(self, inputs, *args):
        x1_batch, x2_batch = inputs[0], inputs[1]
        l2_jacob = self.compute_jacobian(x2_batch)
        e2 = self._f2(x2_batch) - self._f1(x1_batch)
        entropy = self.compute_entropy(e2)
        loss = entropy - self.log(self.abs(l2_jacob)).sum()
        loss = self.get_loss(loss)

        return loss


class PNLWithLossCell(nn.Cell):
    """loss function with network"""

    def __init__(self, loss):
        super(PNLWithLossCell, self).__init__(auto_prefix=False)
        self._loss = loss

    def construct(self, inputs, *args):
        output = self._loss(inputs)

        return output


class MLP(nn.Cell):
    """
    Multi-layer perceptron

    """

    def __init__(self, input_dim, hidden_layers, hidden_units, output_dim,
                 bias=True, activation=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation

        mlp = []
        for i in range(self.hidden_layers):
            input_size = self.hidden_units
            if i == 0:
                input_size = self.input_dim
            weight = nn.Dense(input_size,
                              self.hidden_units,
                              has_bias=self.bias)
            mlp.append(weight)
            if self.activation is not None:
                mlp.append(self.activation)
        out_layer = nn.Dense(self.hidden_units,
                             self.output_dim,
                             has_bias=self.bias)
        mlp.append(out_layer)

        self.mlp = nn.SequentialCell(*mlp)

    def construct(self, x):
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
    hidden_layers: int, default: 1
        number of hidden layer of mlp
    hidden_units: int, default: 10
        number of unit of per hidden layer
    batch_size: int, default: 64
        size of training batch
    iterations: int, default: 100
        training times on all samples
    lr: float, default: 1e-4
        learning rate
    alpha: float, default: 0.05
        significance level
    bias: bool, default: True
        whether use bias
    activation: callable
        nonlinear activation function
    seed: int, default: 42
        random seed

    Examples
    --------
    >>> import castle
    >>> castle.set_backend('mindspore')
    >>> from castle.algorithms import PNL
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = PNL()
    # If you want to use GPU, you can call CORL().set_context() like the following code:
    >>> algo.set_context(device_target='GPU', mode=ms.GRAPH_MODE) # 'GPU', 'CPU' or 'Ascend'
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    def __init__(self, hidden_layers=1, hidden_units=10, batch_size=64,
                 iterations=100, lr=1e-4, alpha=0.05, bias=True,
                 activation=nn.LeakyReLU(alpha=0.01), seed=42):
        super(PNL, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.iterations = iterations
        self.lr = lr
        self.alpha = alpha
        self.bias = bias
        self.activation = activation
        self.seed = seed

        self._set_seed(self.seed)

    def learn(self, data, columns=None, **kwargs):
        """Set up and run the PNL algorithm

        Parameters
        ----------
        data: numpy.ndarray
        columns : Index or array-like, default: None
            Column labels to use for describe nodes.
        """
        x = Tensor(data, columns=columns)
        n_nodes = x.shape[1]
        g = np.zeros((n_nodes, n_nodes))
        all_nodes_pair = itertools.permutations(range(n_nodes), 2)
        progress_bar = tqdm([*all_nodes_pair], disable=False)
        for i, j in progress_bar:
            x1 = x.data[:, i].reshape(x.shape[0], 1).astype(np.float32)
            x2 = x.data[:, j].reshape(x.shape[0], 1).astype(np.float32)
            if columns:
                column_names = [columns[i], columns[j]]
            else:
                column_names = None
            dataset = ds.NumpySlicesDataset(data=(x1, x2), shuffle=False,
                                            column_names=column_names)
            dataset = dataset.batch(batch_size=64)
            iterator = dataset.create_tuple_iterator()

            # initialize model and parameters
            l1 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation)
            l2 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation)
            l1.update_parameters_name("l1")
            l2.update_parameters_name("l2")
            optimizer = nn.SGD([{'params': l1.trainable_params()},
                                {'params': l2.trainable_params()}],
                               learning_rate=self.lr)

            loss = PNLLoss(l1, l2)
            loss_net = PNLWithLossCell(loss)
            train_cell = nn.TrainOneStepCell(loss_net, optimizer)
            for e in range(self.iterations):
                for x1_batch, x2_batch in iterator:
                    train_cell((x1_batch, x2_batch))
                progress_bar.set_description(
                    f"Node {i}-{j} and Inner Epoch {e}/{self.iterations} is training", refresh=True)
            e2 = l2(ms.Tensor(x2, ms.float32)) - l1(ms.Tensor(x1, ms.float32))

            # kernel-based independent test
            ind = hsic_test(ms.Tensor(x1).asnumpy(),
                            e2.asnumpy(), alpha=self.alpha)
            if ind:  # x1->x2
                g[i, j] = 1

            gc.collect()

        self.causal_matrix = Tensor(g, index=x.columns, columns=x.columns)
