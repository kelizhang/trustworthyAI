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


class GradNetWrtX(nn.Cell):
    """
    Calculates the derivative of output to the input

    Parameters
    ----------
    network: instantiating model object
        'NonLinGaussANM' or 'NonLinGauss'
    """
    def __init__(self, network):
        super(GradNetWrtX, self).__init__(auto_prefix=False)
        self.network = network
        self.grad_op = ops.GradOperation(sens_param=True)
        self.flag = ms.Tensor(True, dtype=ms.bool_)

    def construct(self, x, grad_output):
        """
        If you want to calculate the derivative of a single output
        (for example, output[0][0]) to the input,
        you can set the scaling value of the corresponding position to 1
        and set other values to 0.

        Parameters
        ----------
        x: mindspore.Tensor
        grad_output: mindspore.Tensor, have same shape with x

        Returns
        -------
        gout: gradient value, have same shape with x
        """
        gradient_function = self.grad_op(self.network)
        gout = gradient_function(x, self.flag, grad_output)
        return gout


def is_acyclic(adjacency):
    """
    Whether the adjacency matrix is a acyclic graph.
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency.asnumpy(), prod)
        if np.trace(prod) != 0:
            return False
    return True


def compute_jacobian_avg(model, data_manager, batch_size):
    """
    compute the average Jacobian of learned model
    """
    jac_avg = msnp.zeros((model.input_dim, model.input_dim))

    # sample
    x, _ = data_manager.sample(batch_size)
    model.set_train(False)

    # compute jacobian of the loss
    for k in range(model.input_dim):
        grad_output = msnp.zeros(shape=x.shape)
        grad_output[:, k] = 1
        tmp_grad = GradNetWrtX(model)(x, grad_output)
        jac_avg[k, :] = msnp.mean(msnp.absolute(tmp_grad), 0)

    return jac_avg
