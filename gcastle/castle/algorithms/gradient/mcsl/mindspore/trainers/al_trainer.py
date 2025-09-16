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
import numpy as np

import mindspore
import mindspore.nn as nn

from castle.common.consts import LOG_FREQUENCY
from ..helpers.utils import compute_acyclicity, convert_logits_to_sigmoid


class CustomTrainOneStepCell(nn.Cell):
    """Custom Training Network"""

    def __init__(self, network, optimizer):
        """There are two input parameters: training network and optimizer."""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                    # Defining a Forward Network
        self.network.set_grad()                   # Build a reverse network
        self.optimizer = optimizer                # Define Optimizer
        self.weights = self.optimizer.parameters  # Parameters to be updated
        self.grad = mindspore.ops.GradOperation(get_by_list=True)  # Backpropagation Obtains Gradients

    def construct(self, x, rho, alpha, temperature):
        # Execute the forward network and calculate the current input loss function value.
        curr_loss, curr_h, curr_w, mse_loss = self.network(x, rho, alpha, temperature)

        # Perform backpropagation and calculate the gradient.
        grads = self.grad(self.network, self.weights)(x, rho, alpha, temperature)
        loss = mindspore.ops.depend(curr_loss, self.optimizer(grads))  # Updating Gradients Using the Optimizer.
        return loss, curr_h, curr_w, mse_loss


class Trainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """

    def __init__(self, model, learning_rate, init_rho, rho_thresh, h_thresh,
                 rho_multiply, init_iter, h_tol, temperature) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.init_rho = mindspore.Tensor(init_rho)
        self.rho_thresh = rho_thresh
        self.h_thresh = h_thresh
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.temperature = mindspore.Tensor(temperature)
        self.optimizer = nn.Adam(self.model.trainable_params(),
                                 learning_rate=self.learning_rate)

        self.train_one_step = CustomTrainOneStepCell(self.model, self.optimizer)

    def train(self, x, iterations, iter_step) -> mindspore.Tensor:
        """"""

        rho = self.init_rho
        h = np.inf
        h_new = np.inf
        alpha = 0.0
        w_logits_new = None
        for i in range(1, iterations + 1):
            logging.info(f'Current epoch: {i}==================')
            while rho < self.rho_thresh:
                loss_new, h_new, w_logits_new = self.train_step(
                    x, iter_step, rho, alpha, self.temperature
                )
                if h_new > self.h_thresh * h:
                    rho *= self.rho_multiply
                else:
                    break
            # Use two stopping criterion
            h_logits = compute_acyclicity(
                convert_logits_to_sigmoid(w_logits_new,
                                          tau=self.temperature)
            )
            logging.info(f'Current        h: {h_new}')
            logging.info(f'Current h_logits: {h_logits}')
            if (h_new <= self.h_tol
                    # and h_logits <= self.h_tol
                    and i > self.init_iter):
                break
            # Update h and alpha
            h = h_new
            alpha += rho * h

        return w_logits_new

    def train_step(self, x, iter_step, rho, alpha, temperature) -> tuple:
        self.train_one_step.set_train()
        # curr_loss, curr_mse and curr_h are single-sample estimation
        curr_loss, curr_h, curr_w = None, None, None
        for _ in range(iter_step):

            curr_loss, curr_h, curr_w, mse_loss = self.train_one_step(x, rho, alpha, temperature)

            if _ % LOG_FREQUENCY == 0:
                logging.info(f'Current loss in step {_}: {curr_loss}')
                logging.info(f'Current mse_loss in step {_}: {mse_loss}')
        return curr_loss, curr_h, curr_w
