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

import logging
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
import mindspore.ops as ops

from .encoder import TransformerEncoder, GATEncoder
from .decoder import TransformerDecoder, SingleLayerDecoder, BilinearDecoder, NTNDecoder
from .critic import Critic


class Actor(nn.Cell):
    _logger = logging.getLogger(__name__)

    def __init__(self,
                 hidden_dim,
                 max_length,
                 num_heads,
                 num_stacks,
                 residual,
                 encoder_type,
                 decoder_type,
                 decoder_activation,
                 decoder_hidden_dim,
                 use_bias,
                 use_bias_constant,
                 bias_initial_value,
                 batch_size,
                 input_dimension,
                 lr1_start,
                 lr1_decay_step,
                 lr1_decay_rate,
                 alpha,
                 init_baseline):
        super(Actor, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.residual = residual
        self.alpha = alpha
        self.init_baseline = init_baseline
        self.lr1_start = lr1_start
        self.lr1_decay_rate = lr1_decay_rate
        self.lr1_decay_step = lr1_decay_step
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.decoder_activation = decoder_activation
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_bias = use_bias
        self.use_bias_constant = use_bias_constant
        self.bias_initial_value = bias_initial_value
        # Reward config
        self.avg_baseline = ms.Parameter(ms.Tensor([self.init_baseline]),
                                         requires_grad=False)  # moving baseline for Reinforce

        # encoder
        if self.encoder_type == 'TransformerEncoder':
            self.encoder = TransformerEncoder(batch_size=self.batch_size,
                                              max_length=self.max_length,
                                              input_dimension=self.input_dimension,
                                              hidden_dim=self.hidden_dim,
                                              num_heads=self.num_heads,
                                              num_stacks=self.num_stacks)
        elif self.encoder_type == 'GATEncoder':
            self.encoder = GATEncoder(batch_size=self.batch_size,
                                      max_length=self.max_length,
                                      input_dimension=self.input_dimension,
                                      hidden_dim=self.hidden_dim,
                                      num_heads=self.num_heads,
                                      num_stacks=self.num_stacks,
                                      residual=self.residual)
        else:
            raise NotImplementedError('Current encoder type is not implemented yet!')

        # decoder
        if self.decoder_type == 'SingleLayerDecoder':
            self.decoder = SingleLayerDecoder(
                batch_size=self.batch_size,
                max_length=self.max_length,
                input_dimension=self.input_dimension,
                input_embed=self.hidden_dim,
                decoder_hidden_dim=self.decoder_hidden_dim,
                decoder_activation=self.decoder_activation,
                use_bias=self.use_bias,
                bias_initial_value=self.bias_initial_value,
                use_bias_constant=self.use_bias_constant)
        elif self.decoder_type == 'TransformerDecoder':
            self.decoder = TransformerDecoder(batch_size=self.batch_size,
                                              max_length=self.max_length,
                                              hidden_dim=self.hidden_dim,
                                              num_heads=self.num_heads,
                                              num_stacks=self.num_stacks)
        elif self.decoder_type == 'BilinearDecoder':
            self.decoder = BilinearDecoder(batch_size=self.batch_size,
                                           max_length=self.max_length,
                                           hidden_dim=self.hidden_dim,
                                           use_bias=self.use_bias,
                                           bias_initial_value=self.bias_initial_value,
                                           use_bias_constant=self.use_bias_constant)
        elif self.decoder_type == 'NTNDecoder':
            self.decoder = NTNDecoder(batch_size=self.batch_size,
                                      max_length=self.max_length,
                                      hidden_dim=self.hidden_dim,
                                      decoder_hidden_dim=self.decoder_hidden_dim,
                                      decoder_activation=self.decoder_activation,
                                      use_bias=self.use_bias,
                                      bias_initial_value=self.bias_initial_value,
                                      use_bias_constant=self.use_bias_constant)
        else:
            raise NotImplementedError('Current decoder type is not implemented yet!')

        # critic
        self.critic = Critic(batch_size=self.batch_size,
                             max_length=self.max_length,
                             input_dimension=self.input_dimension,
                             hidden_dim=self.hidden_dim,
                             init_baseline=self.init_baseline)
        
        # Optimizer
        self.lr1_scheduler = nn.ExponentialDecayLR(learning_rate=self.lr1_start,
                                                   decay_rate=pow(self.lr1_decay_rate, 1/self.lr1_decay_step),
                                                   decay_steps=1)
        self.opt1 = nn.optim.Adam([
                        {'params': self.encoder.trainable_params()},
                        {'params': self.decoder.trainable_params()},
                        {'params': self.critic.trainable_params()}
                    ], learning_rate=self.lr1_scheduler, beta1=0.9, beta2=0.99, eps=0.0000001)
        
        self.criterion = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def build_permutation(self, inputs):

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        # encoder
        encoder_output = self.encoder(inputs)

        # decoder
        samples, scores, entropy = self.decoder(encoder_output)

        # samples is seq_length * batch size * seq_length
        # cal cross entropy loss * reward
        graphs_gen = samples.transpose([1, 0, 2])
        logits_for_rewards = scores
        logits_for_rewards = logits_for_rewards.transpose([1, 0, 2])
        log_probss = self.bce_loss(logits_for_rewards, graphs_gen)
        log_softmax = ops.mean(log_probss, axis=[1, 2])

        entropy_for_rewards = entropy
        entropy_for_rewards = entropy_for_rewards.transpose([1, 0, 2])
        entropy_regularization = ops.mean(entropy_for_rewards, axis=[1, 2])

        # build critic
        critic_predictions = self.build_critic(encoder_output)

        return graphs_gen, log_softmax, entropy_regularization, critic_predictions

    def build_critic(self, encoder_output):
        predictions_ret = self.critic(encoder_output)
        return predictions_ret

    def build_reward(self, reward, log_softmax, entropy_regularization, critic_predictions):
        loss = self.build_optim(reward, log_softmax, entropy_regularization, critic_predictions)

        return loss

    def build_optim(self, reward, log_softmax, entropy_regularization, critic_predictions):
        # Update moving_mean and moving_variance for batch normalization layers
        # Update baseline
        reward_mean, reward_var = msnp.mean(reward), msnp.std(reward)
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean

        # Discounted reward
        reward_baseline = reward - self.avg_baseline - critic_predictions  # [Batch size, 1]

        # Loss
        loss1 = (msnp.mean(reward_baseline * log_softmax, 0)
                 - 1*self.lr1_scheduler.learning_rate * msnp.mean(entropy_regularization, 0))
        loss2 = self.criterion(reward - self.avg_baseline, critic_predictions)

        loss = loss1 + loss2

        return loss
