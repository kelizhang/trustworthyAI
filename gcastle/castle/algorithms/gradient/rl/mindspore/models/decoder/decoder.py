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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp
import mindspore.nn.probability.distribution as msd
from mindspore import Parameter
from mindspore.common.initializer import initializer, Normal, XavierUniform


class BilinearDecoder(nn.Cell):

    def __init__(self, batch_size, max_length, hidden_dim, use_bias,
                 bias_initial_value, use_bias_constant):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim    # dimension of embedding space (actor)
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant

        self.bernoulli = msd.Bernoulli(dtype=ms.float32)

        self._W = Parameter(ms.Tensor(shape=(self.input_embed, self.input_embed),
                                      dtype=ms.float32, init=Normal()), requires_grad=True)
        if self.use_bias:
            self._l = Parameter(ms.Tensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self._W = initializer(XavierUniform(), shape=self._W.shape)  # variables initializer

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = np.random.randn(1)[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        if self.use_bias:
            self._l = Parameter(ms.Tensor(bias_initial_value, dtype=ms.float32),
                                requires_grad=True)

    def construct(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]

        einsum_one = ops.matmul(encoder_output, self._W)
        logits = ops.matmul(einsum_one, encoder_output.transpose([0, 2, 1]))

        if self.use_bias:  # Bias to control sparsity/density
            logit_bias = self._l
            logits += logit_bias

        mask_scores = []
        prob_val_list = list()
        for i in range(self.max_length):
            mask = msnp.zeros((encoder_output.shape[0], self.max_length))
            mask[:, i] = 1.0
            masked_score = logits[:, i, :] - 1000000. * mask
            masked_score = ops.clip_by_value(masked_score, ms.Tensor(-10.0), ms.Tensor(10.))
            prob_val = 1. - 1. / (1. + ops.exp(masked_score))
            prob_val_list.append(prob_val)
            mask_scores.append(masked_score)

        prob_val = ops.stack(prob_val_list, axis=0)
        entropy = self.bernoulli.entropy(probs1=prob_val)
        samples = self.bernoulli.sample(probs1=prob_val)
        mask_scores = ops.stack(mask_scores)

        return samples, mask_scores, entropy


class NTNDecoder(nn.Cell):

    def __init__(self, batch_size, max_length, hidden_dim,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant):
        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.max_length = max_length
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant

        if self.decoder_activation == 'tanh':  # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self.bernoulli = msd.Bernoulli(dtype=ms.float32)

        self._w = Parameter(ms.Tensor(shape=(self.input_embed, self.input_embed, self.decoder_hidden_dim),
                                      dtype=ms.float32, init=Normal()), requires_grad=True)
        self._wl = Parameter(ms.Tensor(shape=(self.input_embed, self.decoder_hidden_dim),
                                       dtype=ms.float32, init=Normal()), requires_grad=True)
        self._wr = Parameter(ms.Tensor(shape=(self.input_embed, self.decoder_hidden_dim),
                                       dtype=ms.float32, init=Normal()), requires_grad=True)
        self._u = Parameter(ms.Tensor(shape=(self.decoder_hidden_dim, 1),
                                      dtype=ms.float32, init=Normal()), requires_grad=True)
        self._b = Parameter(ms.Tensor(shape=(self.decoder_hidden_dim, 1),
                                      dtype=ms.float32, init=Normal()), requires_grad=True)
        if self.use_bias:
            self._l = Parameter(ms.Tensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self._w = initializer(XavierUniform(), shape=self._w.shape)
        self._wl = initializer(XavierUniform(), shape=self._wl.shape)
        self._wr = initializer(XavierUniform(), shape=self._wr.shape)
        self._u = initializer(XavierUniform(), shape=self._u.shape)
        self._b = initializer(XavierUniform(), shape=self._b.shape)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = np.random.randn(1)[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        if self.use_bias:
            self._l = Parameter(ms.Tensor(bias_initial_value, dtype=ms.float32),
                                requires_grad=True)

    def construct(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]

        W = self._w
        W_l = self._wl
        W_r = self._wr
        U = self._u
        B = self._b

        # Compute linear output with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        dot_l = ops.matmul(encoder_output, W_l)
        dot_r = ops.matmul(encoder_output, W_r)

        tiled_l = msnp.tile(ops.expand_dims(dot_l, axis=2), (1, 1, self.max_length, 1))
        tiled_r = msnp.tile(ops.expand_dims(dot_r, axis=1), (1, self.max_length, 1, 1))

        linear_sum = tiled_l + tiled_r

        # Compute bilinear product with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        bilinear_product0 = ops.matmul(ops.expand_dims(encoder_output, axis=1), W.transpose([2, 0, 1]))
        bilinear_product0 = bilinear_product0.transpose([0, 2, 3, 1])
        bilinear_product = ops.matmul(ops.expand_dims(encoder_output, axis=1), bilinear_product0)

        if self.decoder_activation == 'tanh':  # Original implementation by paper
            final_sum = self.activation(bilinear_product + linear_sum + B.view(self.decoder_hidden_dim))
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(bilinear_product + linear_sum + B.view(self.decoder_hidden_dim))
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = bilinear_product + linear_sum + B.view(self.decoder_hidden_dim)
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        logits = ops.matmul(final_sum, U.view(self.decoder_hidden_dim))

        if self.use_bias:    # Bias to control sparsity/density
            logit_bias = self._l
            logits += logit_bias

        mask_scores = []
        prob_val_list = list()
        for i in range(self.max_length):
            mask = msnp.zeros((encoder_output.shape[0], self.max_length))
            mask[:, i] = 1.0
            masked_score = logits[:, i, :] - 1000000. * mask
            masked_score = ops.clip_by_value(masked_score, ms.Tensor(-10.0), ms.Tensor(10.))
            prob_val = 1. - 1. / (1. + ops.exp(masked_score))
            prob_val_list.append(prob_val)
            mask_scores.append(masked_score)

        prob_val = ops.stack(prob_val_list, axis=0)
        entropy = self.bernoulli.entropy(probs1=prob_val)
        samples = self.bernoulli.sample(probs1=prob_val)
        mask_scores = ops.stack(mask_scores)

        return samples, mask_scores, entropy


class SingleLayerDecoder(nn.Cell):

    def __init__(self, batch_size, max_length, input_dimension, input_embed,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = input_dimension
        self.input_embed = input_embed    # dimension of embedding space (actor)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self.bernoulli = msd.Bernoulli(dtype=ms.float32)

        self._wl = Parameter(ms.Tensor(shape=(self.input_embed, self.decoder_hidden_dim),
                                       dtype=ms.float32, init=Normal()), requires_grad=True)
        self._wr = Parameter(ms.Tensor(shape=(self.input_embed, self.decoder_hidden_dim),
                                       dtype=ms.float32, init=Normal()), requires_grad=True)
        self._u = Parameter(ms.Tensor(shape=(self.decoder_hidden_dim, 1),
                                      dtype=ms.float32, init=Normal()), requires_grad=True)
        if self.use_bias:
            self._l = Parameter(ms.Tensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self._wl = initializer(XavierUniform(), shape=self._wl.shape)
        self._wr = initializer(XavierUniform(), shape=self._wr.shape)
        self._u = initializer(XavierUniform(), shape=self._u.shape)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = np.random.randn(1)[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        if self.use_bias:
            self._l = Parameter(ms.Tensor(bias_initial_value, dtype=ms.float32),
                                requires_grad=True)

    def construct(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        W_l = self._wl
        W_r = self._wr
        U = self._u

        dot_l = ops.matmul(encoder_output, W_l)
        dot_r = ops.matmul(encoder_output, W_r)

        tiled_l = msnp.tile(ops.expand_dims(dot_l, axis=2), (1, 1, self.max_length, 1))
        tiled_r = msnp.tile(ops.expand_dims(dot_r, axis=1), (1, self.max_length, 1, 1))

        if self.decoder_activation == 'tanh':  # Original implementation by paper
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':  # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = ops.matmul(final_sum, U.view(self.decoder_hidden_dim))  # Readability

        if self.use_bias:  # Bias to control sparsity/density
            logit_bias = self._l
            logits += logit_bias

        mask_scores = []
        prob_val_list = list()
        for i in range(self.max_length):
            mask = msnp.zeros((encoder_output.shape[0], self.max_length))
            mask[:, i] = 1.0
            masked_score = logits[:, i, :] - 1000000.*mask
            masked_score = ops.clip_by_value(masked_score, ms.Tensor(-10.0), ms.Tensor(10.))
            prob_val = 1. - 1. / (1. + ops.exp(masked_score))
            prob_val_list.append(prob_val)
            mask_scores.append(masked_score)

        prob_val = ops.stack(prob_val_list, axis=0)
        entropy = self.bernoulli.entropy(probs1=prob_val)
        samples = self.bernoulli.sample(probs1=prob_val)
        mask_scores = ops.stack(mask_scores)

        return samples, mask_scores, entropy


class MultiheadAttention(nn.Cell):
    
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
        self.bn_layer = nn.BatchNorm2d(input_dimension)  # Number of incoming channels

    def construct(self, inputs, num_heads=16, dropout_rate=0.1):

        inputs = inputs.transpose([0, 2, 1])

        Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
        K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
        V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = ops.concat(msnp.array_split(Q, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
        K_ = ops.concat(msnp.array_split(K, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
        V_ = ops.concat(msnp.array_split(V, int(num_heads), axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]

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
        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]
        outputs = ops.squeeze(outputs, axis=-1)

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

        self.bn_layer = nn.BatchNorm2d(num_units[1])  # Number of incoming channels

    def construct(self, inputs):

        outputs = self.conv1(inputs)
        outputs = ops.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = ops.relu(outputs)

        # Residual connection
        outputs = outputs + inputs

        outputs = ops.expand_dims(outputs, axis=-1)
        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]
        outputs = ops.squeeze(outputs, axis=-1)

        return outputs


class TransformerDecoder(nn.Cell):
 
    def __init__(self, batch_size, max_length, hidden_dim,
                 num_heads, num_stacks):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks

        self.conv1_net = nn.Conv1d(in_channels=self.input_embed,
                                   out_channels=self.input_embed,
                                   kernel_size=1, stride=1,
                                   weight_init='xavier_uniform',
                                   bias_init='uniform',
                                   has_bias=True)

        # Batch Normalization
        self.bn_layer = nn.BatchNorm2d(self.input_dimension)  # Number of incoming channels

        # Bernoulli
        self.bernoulli = msd.Bernoulli(dtype=ms.float32)

        # conv1d
        self.conv1 = nn.Conv1d(in_channels=self.input_embed,
                               out_channels=self.max_length,
                               kernel_size=1,
                               weight_init='xavier_uniform',
                               bias_init='uniform',
                               has_bias=True)

        # attention
        self.multihead_attention = MultiheadAttention(self.input_dimension,
                                                      num_units=self.input_embed)

        # feedforward
        self.feedforward = FeedForward(num_units=[self.input_embed, self.input_embed])

    def construct(self, inputs):

        inputs = inputs.transpose([0, 2, 1])
        # Embed input sequence
        embedded_input = self.conv1_net(inputs)

        # Batch Normalization
        embedded_input = ops.expand_dims(embedded_input, axis=-1)
        enc = ops.squeeze(self.bn_layer(embedded_input), axis=-1)
        
        # Blocks
        for _ in range(self.num_stacks):  # num blocks
            # Multihead Attention
            enc = self.multihead_attention(enc, num_heads=self.num_heads,
                                           dropout_rate=0.0)
            # Feed Forward
            enc = self.feedforward(enc)

        # Readout layer
        adj_prob = self.conv1(enc)
        adj_prob = adj_prob.transpose([0, 2, 1])

        # ######### Initialize process ##########
        # Keep track of visited cities
        mask_scores = []
        prob_val_list = list()
        for i in range(self.max_length):
            mask = msnp.zeros((inputs.shape[0], self.max_length))
            mask[:, i] = 1.0
            masked_score = adj_prob[:, i, :] - 1000000. * mask
            masked_score = ops.clip_by_value(masked_score, ms.Tensor(-10.0), ms.Tensor(10.))
            prob_val = 1. - 1. / (1. + ops.exp(masked_score))
            prob_val_list.append(prob_val)
            mask_scores.append(masked_score)

        prob_val = ops.stack(prob_val_list, axis=0)
        entropy = self.bernoulli.entropy(probs1=prob_val)
        samples = self.bernoulli.sample(probs1=prob_val)
        mask_scores = ops.stack(mask_scores)

        return samples, mask_scores, entropy
