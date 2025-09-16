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
import mindspore as ms
from mindspore import ops
from ._base_network import BaseEncoder


class LSTMEncoder(BaseEncoder):
    """
    Parameters
    ----------
    input_dim: int
        Number of features of input.
    embed_dim: int
        Number of features of hidden layer.
    """

    def __init__(self, input_dim, embed_dim) -> None:
        super(LSTMEncoder, self).__init__(input_dim=input_dim,
                                          embed_dim=embed_dim)
        self.input_dim = input_dim
        self.hidden_dim = embed_dim
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            has_bias=True,
                            batch_first=True
                            )

    def construct(self, x) -> ms.Tensor:
        """

        Parameters
        ----------
        x:
            [Batch Size, Sequence Length, Features]
        """

        x = ops.transpose(x, (0, 2, 1))
        output = ops.transpose(self.embedding(x), (0, 2, 1))
        output, (_, _) = self.lstm(output)

        return output


class MLPEncoder(BaseEncoder):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, embed_dim, hidden_dim) -> None:
        super(MLPEncoder, self).__init__(input_dim=input_dim,
                                         embed_dim=embed_dim,
                                         hidden_dim=hidden_dim)
        self.input_dim = input_dim
        self.embed_dim = embed_dim  # also is output_dim
        self.hidden_dim = hidden_dim

    def construct(self, x) -> ms.Tensor:
        x = ops.transpose(x, (0, 2, 1))
        output = self.embedding(x)
        output = self.feedforward_conv1d(output)
        output = ops.expand_dims(output, axis=-1)
        output = self.bn(output)
        output = ops.squeeze(output, axis=-1)
        output = ops.transpose(output, (0, 2, 1))

        return output


class TransformerEncoder(BaseEncoder):
    """Transformer Encoder"""

    def __init__(self, input_dim, embed_dim, hidden_dim,
                 heads=8, blocks=3) -> None:
        super(TransformerEncoder, self).__init__(input_dim=input_dim,
                                                 embed_dim=embed_dim,
                                                 hidden_dim=hidden_dim)
        self.blocks = blocks
        self.attention = MultiHeadAttention(input_dim=embed_dim,
                                            output_dim=embed_dim,
                                            heads=heads,
                                            dropout_rate=0.0)

    def construct(self, x) -> ms.Tensor:
        x = ops.transpose(x, (0, 2, 1))
        output = ops.transpose(self.embedding(x), (0, 2, 1))
        for i in range(self.blocks):
            enc = self.attention(output)
            enc = ops.transpose(enc, (0, 2, 1))
            output = self.feedforward_conv1d(enc)
            output += enc  # Residual connection
            output = ops.expand_dims(output, axis=-1)
            output = self.bn(output)
            output = ops.squeeze(output, axis=-1)
            output = ops.transpose(output, (0, 2, 1))

        return output


class MultiHeadAttention(nn.Cell):
    """
    Multi head attention mechanism

    Parameters
    ----------
    input_dim: int
        input dimension
    output_dim: int
        output dimension
    heads: int
        head numbers of multi head attention mechanism
    dropout_rate: float, int
        If not 0, append `Dropout` layer on the outputs of each LSTM layer
        except the last layer. Default 0. The range of dropout is (0.0, 1.0).

    """

    def __init__(self, input_dim, output_dim, heads=8, dropout_rate=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.dropout_rate = dropout_rate

        self.w_q = nn.SequentialCell(
            nn.Dense(in_channels=input_dim,
                     out_channels=output_dim,
                     weight_init='xavier_uniform',
                     bias_init='uniform'
                     ),
            nn.ReLU()
        )
        self.w_k = nn.SequentialCell(
            nn.Dense(in_channels=input_dim,
                     out_channels=output_dim,
                     weight_init='xavier_uniform',
                     bias_init='uniform'
                     ),
            nn.ReLU()
        )
        self.w_v = nn.SequentialCell(
            nn.Dense(in_channels=input_dim,
                     out_channels=output_dim,
                     weight_init='xavier_uniform',
                     bias_init='uniform'
                     ),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(num_features=output_dim, use_batch_statistics=True)

    def construct(self, x) -> ms.Tensor:
        Q = self.w_q(x)  # [batch_size, seq_length, n_hidden]
        K = self.w_k(x)
        V = self.w_v(x)

        # Split and concat
        Q_ = ops.Concat()(ops.split(Q, axis=2, output_num=self.heads))
        K_ = ops.Concat()(ops.split(K, axis=2, output_num=self.heads))
        V_ = ops.Concat()(ops.split(V, axis=2, output_num=self.heads))

        # Multiplication # [num_heads*batch_size, seq_length, seq_length]
        output = ops.matmul(Q_, ops.transpose(K_, (0, 2, 1)))

        # Scale
        output = output / (K_.shape[-1] ** 0.5)

        # Activation  # [num_heads*batch_size, seq_length, seq_length]
        output = ops.softmax(output, axis=1)
        # Dropouts
        output = nn.Dropout(keep_prob=(1-self.dropout_rate))(output)
        # Weighted sum # [num_heads*batch_size, seq_length, n_hidden/num_heads]
        output = ops.matmul(output, V_)

        # Restore shape
        output = ops.Concat(axis=2)(ops.split(output, axis=0, output_num=self.heads))
        # [batch_size, seq_length, n_hidden]
        # Residual connection
        output += x  # [batch_size, seq_length, n_hidden]
        out_put_ = ops.transpose(output, (0, 2, 1))
        out_put_ = ops.expand_dims(out_put_, axis=-1)
        out_put_ = self.bn(out_put_)
        out_put_ = ops.squeeze(out_put_, axis=-1)
        output_ = ops.transpose(out_put_, (0, 2, 1))
        return output_
