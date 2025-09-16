# coding=utf-8
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
from castle.common import Tensor, GradientBasedLearner
from castle.common.class_factory import ClassFactory
from .trainers.al_trainer import ALTrainer
from .models.model import AutoEncoder


@ClassFactory.algorithms.register()
class GAE(GradientBasedLearner):
    """
    GAE Algorithm.
    A gradient-based algorithm using graph autoencoder to model non-linear
    causal relationships.

    Parameters
    ----------
    hidden_layers: int, default: 1
        number of hidden layers for encoder and decoder
    hidden_dim: int, default: 4
        hidden size for mlp layer
    activation: callable, default: nn.LeakyReLU(0.05)
        nonlinear functional
    iterations: int, default: 10
        Number of iterations for optimization problem
    update_freq: int, default: 3000
        Number of steps for each iteration
    init_iter: int, default: 3
        Initial iteration to disallow early stopping
    lr: float, default: 1e-3
        learning rate
    alpha: float, default: 0.0
        Lagrange multiplier
    beta: float, default: 2.0
        Multiplication to amplify rho each time
    init_rho: float, default: 1.0
        Initial value for rho
    rho_thresh: float, default: 1e30
        Threshold for rho
    gamma: float, default: 0.25
        Threshold for h
    penalty_lambda: float, default: 0.0
        L1 penalty for sparse graph. Set to 0.0 to disable
    h_thresh: float, default: 1e-8
        Tolerance of optimization problem
    graph_thresh: float, default: 0.3
        Threshold to filter out small values in graph
    early_stopping: bool, default: False
        Whether to use early stopping
    early_stopping_thresh: float, default: 1.0
        Threshold ratio for early stopping
    seed: int, default: 1230
        Reproducibility, must be int

    Examples
    --------
    >>> import castle
    >>> castle.set_backend('mindspore')
    >>> from castle.algorithms import GAE
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = GAE()
    # If you want to use GPU, you can call CORL().set_context() like the following code:
    >>> algo.set_context(device_target='GPU')  # 'GPU', 'CPU' or 'Ascend'
    # Or if your backend is mindspore, and you want to use ms.PYNATIVE_MODE, you can:
    >>> import mindspore as ms
    >>> algo.set_context(device_target='GPU', mode=ms.PYNATIVE_MODE)
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    def __init__(self,
                 hidden_layers=1,
                 hidden_dim=4,
                 activation=None,
                 iterations=10,
                 update_freq=3000,
                 init_iter=3,
                 lr=1e-3,
                 alpha=0.0,
                 beta=2.0,
                 init_rho=1.0,
                 rho_thresh=1e30,
                 gamma=0.25,
                 penalty_lambda=0.0,
                 h_thresh=1e-8,
                 graph_thresh=0.14,
                 early_stopping=False,
                 early_stopping_thresh=1.0,
                 seed=1230):

        super(GAE, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        if activation is None:
            activation = nn.LeakyReLU(0.05)
        self.activation = activation
        self.iterations = iterations
        self.update_freq = update_freq
        self.init_iter = init_iter
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.init_rho = init_rho
        self.rho_thresh = rho_thresh
        self.gamma = gamma
        self.penalty_lambda = penalty_lambda
        self.h_thresh = h_thresh
        self.graph_thresh = graph_thresh
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.seed = seed

    def learn(self, data, columns=None, **kwargs):

        data = Tensor(data, columns=columns).data
        x = ms.Tensor(data, dtype=ms.float32)

        self.n, self.d = x.shape[:2]
        if x.ndim == 2:
            x = x.reshape((self.n, self.d, 1))
            self.input_dim = 1
        elif x.ndim == 3:
            self.input_dim = x.shape[2]
        w_est = self._gae(x).asnumpy()
        self.weight_causal_matrix = Tensor(w_est,
                                           index=columns,
                                           columns=columns)
        causal_matrix = (abs(w_est) > self.graph_thresh).astype(int)
        self.causal_matrix = Tensor(causal_matrix,
                                    index=columns,
                                    columns=columns)

    def _gae(self, x):

        self._set_seed(self.seed)
        model = AutoEncoder(d=self.d,
                            input_dim=self.input_dim,
                            hidden_layers=self.hidden_layers,
                            hidden_dim=self.hidden_dim,
                            activation=self.activation
                            )
        trainer = ALTrainer(n=self.n,
                            d=self.d,
                            model=model,
                            lr=self.lr,
                            init_iter=self.init_iter,
                            alpha=self.alpha,
                            beta=self.beta,
                            rho=self.init_rho,
                            l1_penalty=self.penalty_lambda,
                            rho_thresh=self.rho_thresh,
                            h_thresh=self.h_thresh,  # 1e-8
                            early_stopping=self.early_stopping,
                            early_stopping_thresh=self.early_stopping_thresh,
                            gamma=self.gamma,
                            seed=self.seed)
        w_est = trainer.train(x=x,
                              iterations=self.iterations,
                              update_freq=self.update_freq)
        if w_est is None:
            w_est = ms.numpy.zeros((self.d, self.d))
        else:
            w_est = w_est / ms.numpy.max(abs(w_est))

        return w_est
