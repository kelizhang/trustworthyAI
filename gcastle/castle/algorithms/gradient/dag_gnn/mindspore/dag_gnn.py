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

import logging

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp

from castle.common import GradientBasedLearner, Tensor
from castle.common import consts
from castle.common.validator import check_args_value
from castle.common.class_factory import ClassFactory
from castle.algorithms.gradient.dag_gnn.mindspore.utils import functions as func
from castle.algorithms.gradient.dag_gnn.mindspore.models.modules import Encoder
from castle.algorithms.gradient.dag_gnn.mindspore.models.modules import Decoder
from castle.algorithms.gradient.dag_gnn.mindspore.models.modules import DagGnnLoss
from castle.algorithms.gradient.dag_gnn.mindspore.models.modules import NetWithLossCell


@ClassFactory.algorithms.register()
class DAGGNN(GradientBasedLearner):
    """DAG Structure Learning with Graph Neural Networks

    References
    ----------
    https://arxiv.org/pdf/1904.10098.pdf

    Parameters
    ----------
    encoder_type: str, default: 'mlp'
        choose an encoder, 'mlp' or 'sem'.
    decoder_type: str, default: 'mlp'
        choose a decoder, 'mlp' or 'sem'.
    encoder_hidden: int, default: 64
        MLP encoder hidden layer dimension, just one hidden layer.
    latent_dim: int, default equal to input dimension
        encoder output dimension
    decoder_hidden: int, default: 64
        MLP decoder hidden layer dimension, just one hidden layer.
    iterations: int, default: 300
        train iterations
    k_max_iter: int, default: 1e2
        the max iteration number for searching lambda and c.
    batch_size: int, default: 100
        Sample size of each training batch
    lr: float, default: 3e-3
        learning rate
    lr_decay: int, default: 100
        Period of learning rate decay.
    gamma: float, default: 0.3
        Multiplicative factor of learning rate decay.
    init_lambda_a: float, default: 0.0
        coefficient for DAG constraint h(A).
    init_c_a: float, default: 1.0
        coefficient for absolute value h(A).
    c_a_thresh: float, default: 1e20
        control loop by c_a
    eta: int, default: 10
        use for update c_a, greater equal than 1.
    multiply_h: float, default: 0.25
        use for judge whether update c_a.
    tau_a: float, default: 0.0
        coefficient for L-1 norm of A.
    h_tolerance: float, default: 1e-8
        the tolerance of error of h(A) to zero.
    use_a_connect_loss: bool, default: False
        flag to use A connect loss
    use_a_positiver_loss: bool, default: False
        flag to enforce A must have positive values
    graph_threshold: float, default: 0.3
        threshold for learned adjacency matrix binarization.
        greater equal to graph_threshold denotes has causal relationship.
    optimizer: str, default: 'Adam'
        choose optimizer, 'Adam' or 'SGD'
    seed: int, default: 42
        random seed

    Examples
    --------
    >>> import castle
    >>> castle.set_backend('mindspore')
    >>> from castle.algorithms import DAGGNN
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = DAGGNN()
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

    @check_args_value(consts.GNN_VALID_PARAMS)
    def __init__(self,
                 encoder_type='mlp',
                 decoder_type='mlp',
                 encoder_hidden=64,
                 latent_dim=None,
                 decoder_hidden=64,
                 iterations=300,
                 k_max_iter=1e2,
                 batch_size=100,
                 lr=3e-3,
                 lr_decay=100,
                 gamma=0.3,
                 init_lambda_a=0.0,
                 init_c_a=1.0,
                 c_a_thresh=1e20,
                 eta=10,
                 multiply_h=0.25,
                 tau_a=0.0,
                 h_tolerance=1e-8,
                 use_a_connect_loss=False,
                 use_a_positiver_loss=False,
                 graph_threshold=0.3,
                 optimizer='adam',
                 seed=42
                 ):
        super(DAGGNN, self).__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_hidden = encoder_hidden
        self.latent_dim = latent_dim
        self.decoder_hidden = decoder_hidden
        self.iterations = iterations
        self.k_max_iter = int(k_max_iter)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.init_lambda_a = init_lambda_a
        self.init_c_a = init_c_a
        self.c_a_thresh = c_a_thresh
        self.eta = eta
        self.multiply_h = multiply_h
        self.tau_a = tau_a
        self.h_tolerance = h_tolerance
        self.use_a_connect_loss = use_a_connect_loss
        self.use_a_positiver_loss = use_a_positiver_loss
        self.graph_threshold = graph_threshold
        self.optimizer = optimizer
        self.seed = seed
        self.input_dim = None

    @staticmethod
    def get_train_step(encoder, decoder, loss_fn, optimizer):
        """
        Get TrainOneStepCell object

        Parameters
        ----------
        encoder: The encoder network to wrap.
        decoder: The decoder network to wrap.
        loss_fn: The loss function used to compute loss.
        optimizer: Optimizer for updating the network parameters.

        Returns
        -------
        train_cell: TrainOneStepCell object
        """
        net_loss = NetWithLossCell(encoder, decoder, loss_fn)
        train_cell = nn.TrainOneStepCell(net_loss, optimizer)

        return train_cell

    def learn(self, data, columns=None, **kwargs):
        """Set up and run the DAG_GNN algorithm

        Parameters
        ----------
        data: numpy.ndarray or Tensor (include Tensor.data)
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """
        self._set_seed(self.seed)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)
        self.n_samples, self.n_nodes, self.input_dim = data.shape

        if self.latent_dim is None:
            self.latent_dim = self.input_dim
        train_loader = func.get_dataloader(data, batch_size=self.batch_size)

        # initialize encoder and decoder
        adj_A = msnp.zeros((self.n_nodes, self.n_nodes))
        encoder = Encoder(adj_A=adj_A,
                          input_dim=self.input_dim,
                          hidden_dim=self.encoder_hidden,
                          output_dim=self.latent_dim,
                          encoder_type=self.encoder_type.lower())
        decoder = Decoder(input_dim=self.latent_dim,
                          hidden_dim=self.decoder_hidden,
                          output_dim=self.input_dim,
                          decoder_type=self.decoder_type.lower())

        # initialize optimizer
        lr = func.compute_lr(self.lr, self.init_c_a)
        lr_strategy = nn.ExponentialDecayLR(learning_rate=lr,
                                            decay_rate=self.gamma,
                                            decay_steps=self.lr_decay)
        optimizer = func.get_optimizer(self.optimizer, encoder, decoder, lr_strategy)

        # main training
        c_a = self.init_c_a
        lambda_a = self.init_lambda_a
        h_a_new = ms.Tensor(1.0)
        h_a_old = np.inf
        elbo_loss = np.inf
        best_elbo_loss = np.inf
        origin_a = adj_A
        epoch = 0

        # Package training information
        loss_fn = DagGnnLoss(c_a=c_a,
                             lambda_a=lambda_a,
                             graph_threshold=self.graph_threshold,
                             n_nodes=self.n_nodes,
                             tau_a=self.tau_a,
                             use_a_connect_loss=self.use_a_connect_loss,
                             use_a_positiver_loss=self.use_a_positiver_loss)
        train_cell = self.get_train_step(encoder, decoder, loss_fn, optimizer)
        encoder.set_train(mode=True)

        # start training
        for step_k in range(self.k_max_iter):
            while c_a < self.c_a_thresh:
                for epoch in range(self.iterations):
                    elbo_loss, origin_a = self._train(train_step=train_cell,
                                                      loss_fn=loss_fn,
                                                      train_loader=train_loader)
                    if elbo_loss < best_elbo_loss:
                        best_elbo_loss = elbo_loss

                if elbo_loss > 2 * best_elbo_loss:
                    break

                a_new = origin_a.copy()
                h_a_new = func._h_A(a_new, self.n_nodes)
                if h_a_new.asnumpy().item() > self.multiply_h * h_a_old:
                    c_a *= self.eta  # eta
                    loss_fn.c_a = c_a
                    # update optimizer when c_a changed
                    lr = func.compute_lr(self.lr, c_a)
                    lr_strategy = nn.ExponentialDecayLR(learning_rate=lr,
                                                        decay_rate=self.gamma,
                                                        decay_steps=self.lr_decay)
                    optimizer = func.get_optimizer(self.optimizer, encoder, decoder, lr_strategy)
                    # re-package training information
                    train_cell = self.get_train_step(encoder, decoder, loss_fn, optimizer)
                else:
                    break
            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_a_old = h_a_new.asnumpy().item()
            logging.info(f"Iter: {step_k}, epoch: {epoch}, h_new: {h_a_old}")
            lambda_a += c_a * h_a_new.asnumpy().item()
            loss_fn.lambda_a = lambda_a
            if h_a_old <= self.h_tolerance:
                break

        origin_a = origin_a.asnumpy()
        origin_a[np.abs(origin_a) < self.graph_threshold] = 0
        origin_a[np.abs(origin_a) >= self.graph_threshold] = 1

        self.causal_matrix = Tensor(origin_a, index=columns, columns=columns)

    @staticmethod
    def _train(train_step, loss_fn, train_loader):
        """
        Training model

        Parameters
        ----------
        train_step: TrainOneStepCell object
        loss_fn: The loss function used to compute loss
        train_loader: data loader, GeneratorDataset class object

        Returns
        -------
        mean loss, weighted adjacency matrix
        """
        nll_train = []
        kl_train = []
        origin_a = None
        for data in train_loader.create_tuple_iterator():
            train_x = data[0].astype(ms.float32)
            train_step(train_x)

            origin_a = loss_fn.origin_a
            nll_train.append(loss_fn.loss_nll.asnumpy().item())
            kl_train.append(loss_fn.loss_kl.asnumpy().item())

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), origin_a
