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

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops


class Critic(nn.Cell):

    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 init_baseline):
        super().__init__()

        self.batch_size = batch_size
        self.max_length = max_length
        self.input_dimension = input_dimension
        # Network config
        self.input_embed = hidden_dim
        self.num_neurons = hidden_dim

        # Baseline setup
        self.init_baseline = init_baseline

        layer0 = nn.Dense(in_channels=self.input_dimension, out_channels=self.num_neurons,
                          weight_init='xavier_uniform', bias_init='uniform')
        self.h0_layer = nn.SequentialCell(layer0, nn.ReLU())

        self.layer1 = nn.Dense(in_channels=self.num_neurons, out_channels=1,
                               weight_init='xavier_uniform', bias_init='uniform')
        self.layer1.bias = ms.Tensor([self.init_baseline])

    def construct(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = msnp.mean(encoder_output, axis=1)
 
        # ffn 1
        h0 = self.h0_layer(frame)
        # ffn 2
        h1 = self.layer1(h0)

        predictions = ops.squeeze(h1)

        return predictions
