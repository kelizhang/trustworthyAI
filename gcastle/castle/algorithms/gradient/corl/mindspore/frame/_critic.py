# coding = utf-8
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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform


class EpisodicCritic(nn.Cell):
    """"""

    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU()) -> None:
        super(EpisodicCritic, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.activation = activation

        # trainable parameters
        env_w0 = initializer(XavierUniform(), (self.input_dim, self.neurons[0]), ms.float32)
        self.env_w0 = ms.Parameter(env_w0, requires_grad=True)
        env_w1 = initializer(XavierUniform(), (self.neurons[0], self.neurons[1]), ms.float32)
        self.env_w1 = ms.Parameter(env_w1, requires_grad=True)
        env_w2 = initializer(XavierUniform(), (self.neurons[1], self.neurons[-1]), ms.float32)
        self.env_w2 = ms.Parameter(env_w2, requires_grad=True)

        env_b1 = ms.Tensor([0.])
        self.env_b1 = ms.Parameter(env_b1, requires_grad=True)

        # Un-trainable parameters
        self.tgt_w0 = initializer(XavierUniform(), (self.input_dim, self.neurons[0]), ms.float32)
        self.tgt_w1 = initializer(XavierUniform(), (self.neurons[0], self.neurons[1]), ms.float32)
        self.tgt_w2 = initializer(XavierUniform(), (self.neurons[1], self.neurons[-1]), ms.float32)
        self.tgt_b1 = ms.Tensor([0.])

    def predict_env(self, stats_x) -> None:
        """predict environment reward"""

        stats_x = ops.stop_gradient(stats_x)
        h0 = ops.matmul(stats_x, self.env_w0)
        h0 = self.activation(h0)
        h1 = ops.matmul(h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = ops.matmul(h1, self.env_w2)
        h2 = self.activation(h2)

        # [batch_size, seq_length - 1]
        self.prediction_env = (h2 + self.env_b1).squeeze()

    def predict_tgt(self, stats_y) -> None:
        """predict target reward"""
        stats_y = ops.stop_gradient(stats_y)
        h0 = ops.matmul(stats_y, self.env_w0)
        h0 = self.activation(h0)
        h1 = ops.matmul(h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = ops.matmul(h1, self.env_w2)
        h2 = self.activation(h2)

        self.prediction_tgt = (h2 + self.tgt_b1).squeeze()

    def soft_replacement(self) -> None:
        # soft_replacement
        self.tgt_w0 = 0.95 * self.tgt_w0 + 0.05 * ops.stop_gradient(self.env_w0)
        self.tgt_w1 = 0.95 * self.tgt_w1 + 0.05 * ops.stop_gradient(self.env_w1)
        self.tgt_w2 = 0.95 * self.tgt_w2 + 0.05 * ops.stop_gradient(self.env_w2)
        self.tgt_b1 = 0.95 * self.tgt_b1 + 0.05 * ops.stop_gradient(self.env_b1)


class DenseCritic(nn.Cell):
    """Critic network for `dense reward` type

    Only one layer network.
    """

    def __init__(self, input_dim, output_dim) -> None:
        super(DenseCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.h0 = nn.SequentialCell(
            nn.Dense(in_channels=input_dim,
                     out_channels=output_dim,
                     weight_init='xavier_uniform',
                     bias_init='uniform'
                     ),
            nn.ReLU()
        )
        w1 = ops.UniformReal()((self.output_dim, 1))
        self.w1 = ms.Parameter(w1, requires_grad=True)
        b1 = ms.Tensor([0.])
        self.b1 = ms.Parameter(b1, requires_grad=True)
        self.params = ms.ParameterTuple((self.w1, self.b1))

    def predict_reward(self, encoder_output) -> ms.Tensor:
        """Predict reward for `dense reward` type"""

        frame = ops.ReduceMean(keep_dims=False)(encoder_output, 1)
        frame = ops.stop_gradient(frame)
        h0 = self.h0(frame)
        prediction = ops.matmul(h0, self.w1) + self.b1

        return prediction.squeeze()
