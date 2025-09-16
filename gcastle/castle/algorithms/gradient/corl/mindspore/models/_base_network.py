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


class BaseEncoder(nn.Cell):
    """Base class for network"""

    def __init__(self, input_dim, embed_dim, hidden_dim=1024) -> None:
        super(BaseEncoder, self).__init__()
        if embed_dim is None:
            embed_dim = input_dim

        # this layer just for Encoder
        self.embedding = nn.SequentialCell(
            [nn.Conv1d(in_channels=input_dim,
                       out_channels=embed_dim,
                       kernel_size=1,
                       weight_init='xavier_uniform',
                       bias_init='uniform',
                       pad_mode="pad",
                       has_bias=True)]
        )
        if self.__class__.__name__ in ['TransformerEncoder', 'MLPEncoder']:
            # this layer just for ``TransformerEncoder`` and ``MLPEncoder``.
            self.feedforward_conv1d = nn.SequentialCell(
                nn.Conv1d(in_channels=embed_dim,
                          out_channels=hidden_dim,
                          kernel_size=1,
                          weight_init='xavier_uniform',
                          bias_init='uniform',
                          pad_mode="pad",
                          has_bias=True),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=embed_dim,
                          weight_init='xavier_uniform',
                          bias_init='uniform',
                          pad_mode="pad",
                          kernel_size=1,
                          has_bias=True),
            )
        self.bn = nn.BatchNorm2d(num_features=embed_dim, use_batch_statistics=True)


class BaseDecoder(nn.Cell):

    def __init__(self, input_dim, hidden_dim) -> None:
        # input_dim is embed_dim
        super(BaseDecoder, self).__init__()
        if self.__class__.__name__ == 'MLPDecoder':
            # this layer just for MLPDecoder
            self.feedforward_mlp = nn.SequentialCell(
                nn.Dense(in_channels=input_dim,
                         out_channels=hidden_dim,
                         weight_init='xavier_uniform',
                         bias_init='uniform'
                         ),
                nn.ReLU(),
                nn.Dense(in_channels=hidden_dim,
                         out_channels=input_dim,
                         weight_init='xavier_uniform',
                         bias_init='uniform'
                         )
            )
        # self.bn = nn.BatchNorm2d(num_features=input_dim)


class PointerDecoder(BaseDecoder):
    """Base Class for all Decoder"""

    def __init__(self, input_dim, hidden_dim) -> None:
        super(PointerDecoder, self).__init__(input_dim=input_dim,
                                             hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.positions = []  # store visited cities for reward
        self.mask = ops.Zeros()(1, ms.float32)
        self.mask_scores = []
        # Attention mechanism -- glimpse  _encoder_glimpse
        self.conv1d_ref_g = nn.Conv1d(in_channels=input_dim,
                                      out_channels=hidden_dim,
                                      kernel_size=1,
                                      weight_init='xavier_uniform',
                                      bias_init='uniform',
                                      pad_mode="pad",
                                      has_bias=True)
        self.w_q_g = nn.Dense(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              weight_init='xavier_uniform',
                              bias_init='uniform',
                              has_bias=False)
        self.v_g = nn.Dense(in_channels=hidden_dim,
                            out_channels=1,
                            weight_init='xavier_uniform',
                            bias_init='uniform',
                            has_bias=False)
        # Pointer mechanism  _encoder_pointer
        self.conv1d_ref_p = nn.Conv1d(in_channels=input_dim,
                                      out_channels=hidden_dim,
                                      kernel_size=1,
                                      weight_init='xavier_uniform',
                                      bias_init='uniform',
                                      pad_mode="pad",
                                      has_bias=True)
        self.w_q_p = nn.Dense(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              weight_init='xavier_uniform',
                              bias_init='uniform',
                              has_bias=False)
        self.v_p = nn.Dense(in_channels=hidden_dim,
                           out_channels=1,
                            weight_init='xavier_uniform',
                            bias_init='uniform',
                           has_bias=False)

    def step_decode(self, input, state=None) -> tuple:
        """

        Parameters
        ----------
        input:
            Encoder's output
        state: tuple, None
            (h, c) for Pointer and None for MLP

        Returns
        -------
        output: tuple

        """

        if self.__class__.__name__ == 'LSTMDecoder':
            # Run the cell on a combination of the previous input and state
            h, c = self.lstm_cell(input, state)
            output = h
            state = (h, c)
        elif self.__class__.__name__ == 'MLPDecoder':
            output = self.mlp(input).squeeze()
        else:
            raise TypeError(f'Supported subclass of PointerDecoder is one of '
                            f'[`LSTMDecoder`, `MLPDecoder`], but got'
                            f'``{self.__class__.__name__}``.')

        # [batch_size, time_sequence]
        masked_scores = self.pointer_net(self.encoder_output, output)

        # Multinomial distribution
        prob_val = 1. - 1. / (1. + ops.exp(masked_scores))
        prob = nn.probability.distribution.Categorical()
        action = prob.sample(probs=prob_val).astype(ms.int64)

        self.mask = self.mask + ops.OneHot()(action, self.seq_length, ms.Tensor(1.0, ms.float32),
                                             ms.Tensor(0.0, ms.float32)).astype(ms.int64)

        # Retrieve decoder's new input
        action_index = ms.numpy.tile(action.reshape(-1, 1, 1), (1, 1, self.hidden_dim))
        next_input = ops.gather_elements(self.encoder_output[:, :1, :], 0, action_index)[:, 0, :]

        return next_input, state, action, masked_scores

    def pointer_net(self, ref, query) -> ms.Tensor:
        """Attention mechanism + Pointer mechanism

        Parameters
        ----------
        ref: torch.Tensor
            encoder_states
        query: torch.Tensor
            decoder_states
        """

        # Attention mechanism
        ref_ = ops.transpose(ref, (0, 2, 1))
        encoder_ref_g = self.conv1d_ref_g(ref_)
        encoder_ref_g = ops.transpose(encoder_ref_g, (0, 2, 1))
        encoder_query_g = ops.expand_dims(self.w_q_g(query), axis=1)
        scores_g = ops.mean(self.v_g(ops.tanh(encoder_ref_g + encoder_query_g)),-1, keep_dims=False)

        attention_g = ops.softmax(scores_g - self.mask * 1e9, axis=-1)

        glimpse = ops.mul(ref, ops.expand_dims(attention_g, axis=2))
        glimpse = ops.ReduceSum(keep_dims=False)(glimpse, 1) + query

        # Pointer mechanism
        encoder_ref_p = ops.transpose(self.conv1d_ref_p(ops.transpose(ref, (0, 2, 1))), (0, 2, 1))
        encoder_query_p = ops.expand_dims(self.w_q_p(glimpse), axis=1)
        scores_p = ops.mean(self.v_p(ops.tanh(encoder_ref_p + encoder_query_p)),-1, keep_dims=False)
        if self.__class__.__name__ == 'MLPDecoder':
            scores_p = 10.0 * ops.tanh(scores_p)
        masked_scores = scores_p - self.mask * 1e9

        return masked_scores

    def log_softmax(self, input, position, mask, state_0, state_1) -> ms.Tensor:

        if self.__class__.__name__ == 'LSTMDecoder':
            state = state_0, state_1
            h, c = self.lstm_cell(input, state)
            output = h
        elif self.__class__.__name__ == 'MLPDecoder':
            output = self.mlp(input)
        else:
            raise TypeError(f'Supported subclass of PointerDecoder is one of '
                            f'[`LSTMDecoder`, `MLPDecoder`], but got'
                            f'``{self.__class__.__name__}``.')
        self.mask = ms.Tensor(mask, dtype=ms.float32)

        # encoder_output_expand
        encoder_output_ex = ops.expand_dims(self.encoder_output, axis=1)
        encoder_output_ex = ms.numpy.tile(encoder_output_ex, (1, self.seq_length, 1, 1))
        encoder_output_ex = encoder_output_ex.reshape(-1, self.seq_length, self.hidden_dim)
        masked_scores = self.pointer_net(encoder_output_ex, output)

        masked_scores = ops.clip_by_value(masked_scores, ms.Tensor(-10.0), ms.Tensor(10.))
        prob_val = 1. - 1. / (1. + ops.exp(masked_scores))
        prob = nn.probability.distribution.Categorical()
        log_softmax = prob.log_prob(value=position.reshape(-1, ), probs=prob_val)
        self.mask = ops.Zeros()(1, ms.float32)

        return log_softmax
