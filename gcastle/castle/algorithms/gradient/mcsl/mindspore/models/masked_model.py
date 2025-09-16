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

from ..helpers.utils import gumbel_sigmoid, matrix_exp
from .masked_nn import MaskedNN
from .masked_quadratic_regression import MaskedQuadraticRegression


class MaskedModel(mindspore.nn.Cell):

    def __init__(self, model_type, n_samples, n_nodes, pns_mask, num_hidden_layers,
                 hidden_dim, l1_graph_penalty, seed) -> None:
        super(MaskedModel, self).__init__()
        self.model_type = model_type
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.pns_mask = pns_mask
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.seed = seed

        if self.model_type == 'nn':
            self.masked_model = MaskedNN(mask=self.pns_mask,
                                         num_hidden_layers=self.num_hidden_layers,
                                         hidden_dim=self.hidden_dim,
                                         )
        elif self.model_type == 'qr':  # quadratic regression
            self.masked_model = MaskedQuadraticRegression(
                mask=self.pns_mask,
                n_samples=self.n_samples,
                n_nodes=self.n_nodes,
                seed=self.seed,
            )
        else:
            raise TypeError(f"The argument `model_type` must be one of"
                            f"['nn', 'qr'], but got {self.model_type}.")

        w = mindspore.ops.uniform((self.n_nodes, self.n_nodes), mindspore.Tensor(-1e-10), mindspore.Tensor(1e-10),
                                  seed=self.seed)
        self.w = mindspore.Parameter(w)

    def construct(self, x, rho, alpha, temperature) -> tuple:
        w_prime = self.preprocess_graph(self.w, tau=temperature, seed=self.seed)
        w_prime *= self.pns_mask
        mse_loss = self._get_mse_loss(x, w_prime)
        h = (matrix_exp(w_prime * w_prime).trace() - self.n_nodes)

        if mindspore.__version__ >= '2.0.0':
            w_prime_norm = mindspore.ops.norm(w_prime, dim=(0, 1), ord=1)
        else:
            w_prime_norm = mindspore.ops.norm(w_prime, axis=(0, 1), p=1)
        loss = (0.5 / self.n_samples * mse_loss
                + self.l1_graph_penalty * w_prime_norm
                + alpha * h
                + 0.5 * rho * h * h)

        return loss, h, self.w, mse_loss

    @staticmethod
    def preprocess_graph(w, tau, seed=0) -> mindspore.Tensor:
        w_prob = gumbel_sigmoid(w, temperature=tau, seed=seed)
        w_prob = (1. - mindspore.ops.Eye()(w.shape[0], w.shape[0], mindspore.float32)) * w_prob
        return w_prob

    def _get_mse_loss(self, x, w_prime):
        """
        Different model for different nodes to use masked features to predict
        value for each node.
        """

        mse_loss = mindspore.Tensor(0)
        for i in range(self.n_nodes):
            # Get possible PNS parents and also remove diagonal element
            curr_y = x[:, i]  # Label for current node
            curr_masked_x = x * self.pns_mask[:, i] * w_prime[:, i]  # Broadcasting
            # Use masked features to predict value of current node
            curr_y_pred = self.masked_model(curr_masked_x, choice=i)
            mse_loss += (curr_y_pred.squeeze() - curr_y).square().sum()
        return mse_loss
