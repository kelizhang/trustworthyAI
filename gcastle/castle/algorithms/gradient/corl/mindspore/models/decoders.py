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

import mindspore.nn as nn
from mindspore import ops
import mindspore as ms
from ._base_network import PointerDecoder


class LSTMDecoder(PointerDecoder):
    """LSTM + Pointer Network"""

    def __init__(self, input_dim, hidden_dim) -> None:
        # input of Decoder is output of Encoder, e.g. embed_dim
        super(LSTMDecoder, self).__init__(input_dim=input_dim,
                                          hidden_dim=hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_cell = nn.LSTMCell(input_size=hidden_dim,
                                     hidden_size=hidden_dim)

    def construct(self, x) -> tuple:
        """"""
        self.batch_size = x.shape[0]
        self.seq_length = x.shape[1]
        self.encoder_output = x  # 保存起来有用

        s_i = ops.ReduceMean(keep_dims=False)(x, 1)
        hi_ci = (
                ops.Zeros()((self.batch_size, self.hidden_dim), ms.float32),
                ops.Zeros()((self.batch_size, self.hidden_dim), ms.float32)
                 )
        h_list = []
        c_list = []
        s_list = []
        action_list = []
        prob_list = []
        for step in range(self.seq_length):
            h_list.append(hi_ci[0])
            c_list.append(hi_ci[1])
            s_list.append(s_i)

            s_i, hi_ci, pos, prob = self.step_decode(input=s_i, state=hi_ci)

            action_list.append(pos)
            prob_list.append(prob)

        h_list = ops.stack(h_list, axis=1).squeeze()  # [Batch,seq_length,hidden]
        c_list = ops.stack(c_list, axis=1).squeeze()  # [Batch,seq_length,hidden]
        s_list = ops.stack(s_list, axis=1).squeeze()  # [Batch,seq_length,hidden]

        # Stack visited indices
        actions = ops.stack(action_list, axis=1)  # [Batch,seq_length]
        mask_scores = ops.stack(prob_list, axis=1)  # [Batch,seq_length,seq_length]
        self.mask = ops.Zeros()(1, ms.float32)

        return actions, mask_scores, s_list, h_list, c_list


class MLPDecoder(PointerDecoder):
    """Multi Layer Perceptions + Pointer Network"""

    def __init__(self, input_dim, hidden_dim) -> None:
        super(MLPDecoder, self).__init__(input_dim=input_dim,
                                         hidden_dim=hidden_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = self.feedforward_mlp

    def construct(self, x) -> tuple:

        self.batch_size = x.shape[0]
        self.seq_length = x.shape[1]
        self.encoder_output = x

        s_i = ops.ReduceMean(keep_dims=False)(x, 1)

        s_list = []
        action_list = []
        prob_list = []
        for step in range(self.seq_length):
            s_list.append(s_i)
            s_i, _, pos, prob = self.step_decode(input=s_i, state=None)

            action_list.append(pos)
            prob_list.append(prob)
        s_list = ops.stack(s_list, axis=1).squeeze()  # [Batch,seq_length,hidden]
        # Stack visited indices
        actions = ops.stack(action_list, axis=1)   # [Batch,seq_length]
        mask_scores = ops.stack(prob_list, axis=1)  # [Batch,seq_length,seq_length]
        self.mask = ops.Zeros()(1, ms.float32)

        return actions, mask_scores, s_list, s_list, s_list
