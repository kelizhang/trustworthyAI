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
import mindspore.ops as ops
from mindspore.common.initializer import initializer, Uniform


class MLP(nn.Cell):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, layers, units, output_dim, activation=None) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.layers = layers
        self.units = units
        self.output_dim = output_dim
        self.activation = activation

        mlp = []
        for i in range(self.layers):
            input_size = units
            if i == 0:
                input_size = input_dim
            weight = nn.Dense(in_channels=input_size,
                              out_channels=self.units,
                              has_bias=True,
                              activation=self.activation)
            mlp.append(weight)
        out_layer = nn.Dense(in_channels=self.units,
                             out_channels=self.output_dim,
                             has_bias=True)
        mlp.append(out_layer)

        self.mlp = nn.SequentialCell(*mlp)

    def construct(self, x) -> ms.Tensor:

        x_ = x.reshape(-1, self.input_dim)
        output = self.mlp(x_)

        return output.reshape(x.shape[0], -1, self.output_dim)


class AutoEncoder(nn.Cell):

    def __init__(self, d, input_dim, hidden_layers=3, hidden_dim=16, activation=None):
        super(AutoEncoder, self).__init__()
        self.d = d
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation

        self.encoder = MLP(input_dim=self.input_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.hidden_dim,
                           activation=self.activation)
        self.decoder = MLP(input_dim=self.hidden_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.input_dim,
                           activation=self.activation)

        w_adj = initializer(Uniform(scale=0.1), (self.d, self.d,), dtype=ms.float32)
        self.w_adj = ms.Parameter(w_adj, requires_grad=True)

    def construct(self, x):

        w_adj = self._preprocess_graph(self.w_adj)
        out = self.encoder(x)
        out = ops.transpose(out, (2, 0, 1))
        out = ops.matmul(out, w_adj)
        out = ops.transpose(out, (1, 2, 0))
        x_est = self.decoder(out)

        return x_est, w_adj

    def _preprocess_graph(self, w_adj):

        return (1. - ops.eye(*w_adj.shape, ms.float32)) * w_adj
