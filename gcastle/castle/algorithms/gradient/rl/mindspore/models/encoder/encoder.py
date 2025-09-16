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
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore import Parameter


class AttnHead(nn.Cell):

    def __init__(self, hidden_dim, out_sz, residual):

        super().__init__()

        self.residual = residual

        self._seq_fts = nn.Conv1d(in_channels=hidden_dim,
                                  out_channels=out_sz,
                                  kernel_size=1,
                                  weight_init='xavier_uniform',
                                  bias_init='uniform',
                                  has_bias=False)

        self._f_1 = nn.Conv1d(in_channels=out_sz,
                              out_channels=1,
                              kernel_size=1,
                              weight_init='xavier_uniform',
                              bias_init='uniform',
                              has_bias=True)

        self._f_2 = nn.Conv1d(in_channels=out_sz,
                              out_channels=1,
                              kernel_size=1,
                              weight_init='xavier_uniform',
                              bias_init='uniform',
                              has_bias=True)

        if self.residual:
            self._ret = nn.Conv1d(in_channels=hidden_dim,
                                  out_channels=out_sz,
                                  kernel_size=1,
                                  weight_init='xavier_uniform',
                                  bias_init='uniform',
                                  has_bias=True)

        self._bias = Parameter(msnp.zeros(1), requires_grad=True)

    def construct(self, seq, in_drop=0.0, coef_drop=0.0, residual=False):

        seq = seq.transpose([0, 2, 1])

        if in_drop != 0.0:
            seq, _ = ops.dropout(seq, p=in_drop)

        seq_fts = self._seq_fts(seq)

        # simplest self-attention possible
        f_1 = self._f_1(seq_fts)
        f_2 = self._f_2(seq_fts)

        logits = f_1 + f_2.transpose([0, 2, 1])
        coefs = ops.softmax(nn.LeakyReLU()(logits), axis=0)

        if coef_drop != 0.0:
            coefs, _ = ops.dropout(coefs, coef_drop)
        if in_drop != 0.0:
            seq_fts, _ = ops.dropout(seq_fts, in_drop)

        vals = ops.matmul(coefs.transpose([0, 2, 1]), seq_fts.transpose([0, 2, 1]))
        ret = vals + self._bias

        # residual connection
        if residual:
            if seq.transpose([0, 2, 1]).shape[-1] != ret.shape[-1]:
                ret = ret + self._ret(seq).transpose([0, 2, 1])  # activation
            else:
                ret = ret + seq.transpose([0, 2, 1])

        return ops.Elu()(ret)  # activation


class GATEncoder(nn.Cell):
 
    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 num_heads, num_stacks, residual):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        self.input_dimension = input_dimension  # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
 
        self.hidden_dim = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.residual = residual
        self.head_hidden_dim = int(self.hidden_dim / self.num_heads)

        self.attn_head = AttnHead(self.hidden_dim, self.head_hidden_dim, self.residual)

    def construct(self, inputs):
        """
        input shape: (batch_size, max_length, input_dimension)
        output shape: (batch_size, max_length, input_embed)
        """
        # First stack
        h_1 = inputs
        for _ in range(self.num_stacks):
            attns = []
            for _ in range(self.num_heads):
                attns.append(self.attn_head(h_1,  in_drop=0, coef_drop=0, residual=self.residual))
            h_1 = ops.concat(attns, axis=-1)

        return h_1


class MultiheadAttention(nn.Cell):
    """
    Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
    Attention size = n_hidden should be a multiple of num_head
    Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
	
    Adapted from kyubyong park, June 2017.
    kbpark.linguist@gmail.com.
    https://www.github.com/kyubyong/transformer
    """
    
    def __init__(self, input_dimension, num_units=None):

        super().__init__()
        # Linear projections
        self.Q_layer = nn.SequentialCell(nn.Dense(in_channels=input_dimension,
                                                  out_channels=num_units,
                                                  weight_init='xavier_uniform',
                                                  bias_init='uniform'),
                                         nn.ReLU())

        self.K_layer = nn.SequentialCell(nn.Dense(in_channels=input_dimension,
                                                  out_channels=num_units,
                                                  weight_init='xavier_uniform',
                                                  bias_init='uniform'),
                                         nn.ReLU())

        self.V_layer = nn.SequentialCell(nn.Dense(in_channels=input_dimension,
                                                  out_channels=num_units,
                                                  weight_init='xavier_uniform',
                                                  bias_init='uniform'),
                                         nn.ReLU())
        
        # Normalize
        self.bn_layer = nn.BatchNorm2d(input_dimension)

    def construct(self, inputs, num_heads=16, dropout_rate=0.1):

        inputs = inputs.transpose([0, 2, 1])

        Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
        K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
        V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = ops.concat(msnp.array_split(Q, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, num_heads]
        K_ = ops.concat(msnp.array_split(K, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, num_heads]
        V_ = ops.concat(msnp.array_split(V, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, num_heads]

        # Multiplication
        outputs = ops.matmul(Q_, K_.transpose([0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]
        
        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Activation
        outputs = ops.softmax(outputs, axis=0)  # num_heads*[batch_size, seq_length, seq_length]

        # Dropouts
        outputs, _ = ops.dropout(outputs, p=dropout_rate)

        # Weighted sum
        outputs = ops.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = ops.concat(msnp.array_split(outputs, int(num_heads), axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
        
        # Residual connection
        outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

        outputs = outputs.transpose([0, 2, 1])
        
        # Normalize
        outputs = ops.expand_dims(outputs, axis=-1)
        outputs = self.bn_layer(outputs)
        outputs = ops.squeeze(outputs, axis=-1)  # [batch_size, seq_length, n_hidden]

        return outputs


class FeedForward(nn.Cell):
    """
    Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
    Returns: a 3d tensor with the same shape and dtype as inputs
    """

    def __init__(self, num_units=(2048, 512)):

        super().__init__()
        # Inner layer
        self.conv1 = nn.Conv1d(in_channels=num_units[1],
                               out_channels=num_units[0],
                               kernel_size=1,
                               weight_init='xavier_uniform',
                               bias_init='uniform',
                               has_bias=True)
        # Readout layer
        self.conv2 = nn.Conv1d(in_channels=num_units[0],
                               out_channels=num_units[1],
                               kernel_size=1,
                               weight_init='xavier_uniform',
                               bias_init='uniform',
                               has_bias=True)

        self.bn_layer1 = nn.BatchNorm2d(num_units[1])  # Number of incoming channels

    def construct(self, inputs):

        outputs = self.conv1(inputs)
        outputs = ops.relu(outputs)
        outputs = self.conv2(outputs)

        # Residual connection
        outputs = outputs + inputs

        outputs = ops.expand_dims(outputs, axis=-1)
        outputs = self.bn_layer1(outputs)
        outputs = ops.squeeze(outputs, axis=-1)  # [batch_size, seq_length, n_hidden]

        return outputs


class TransformerEncoder(nn.Cell):
 
    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 num_heads, num_stacks):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_dimension = input_dimension

        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks

        # Conv1d
        self.conv1d_net = nn.Conv1d(in_channels=self.input_dimension,
                                    out_channels=self.input_embed,
                                    kernel_size=1, stride=1,
                                    weight_init='xavier_uniform',
                                    bias_init='uniform',
                                    has_bias=True)

        # Batch Normalization
        self.bn_layer2 = nn.BatchNorm2d(self.input_dimension)

        # attention
        self.multihead_attention = []
        for i in range(self.num_stacks):  # num blocks
            multihead_attention = MultiheadAttention(self.input_dimension,
                                                     num_units=self.input_embed)
            multihead_attention.update_parameters_name(prefix='attention{}.'.format(i))
            self.multihead_attention.append(multihead_attention)

        # FeedForward
        self.feedforward = []
        for i in range(self.num_stacks):  # num blocks
            feedforward = FeedForward(num_units=(4*self.input_embed, self.input_embed))
            feedforward.update_parameters_name(prefix='feedforward{}.'.format(i))
            self.feedforward.append(feedforward)

    def construct(self, inputs):

        inputs = inputs.transpose([0, 2, 1])
        embedded_input = self.conv1d_net(inputs)
        # Batch Normalization
        embedded_input = ops.expand_dims(embedded_input, axis=-1)
        enc = ops.squeeze(self.bn_layer2(embedded_input), axis=-1)
        
        # Blocks
        for i in range(self.num_stacks):  # num blocks
            enc = self.multihead_attention[i](enc, num_heads=self.num_heads,
                                              dropout_rate=0.0)
            # Feed Forward
            enc = self.feedforward[i](enc)

        # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
        encoder_output = enc  # NOTE: encoder_output is the ref for attention
        encoder_output = encoder_output.transpose([0, 2, 1])

        return encoder_output
