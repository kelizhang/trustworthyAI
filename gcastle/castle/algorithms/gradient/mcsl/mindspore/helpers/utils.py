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

from mindspore import Tensor
from mindspore import ops, nn, numpy as msnp, dtype as mstype


def convert_logits_to_sigmoid(w, tau=1.0):
    d = w.shape[0]
    sigmoid_w = ops.Sigmoid()(w / tau)
    sigmoid_w = sigmoid_w * (1. - ops.Eye()(d, d, mstype.float32))  # Mask diagonal to 0
    return sigmoid_w


def compute_acyclicity(w):
    """

    Parameters
    ----------
    w: Tensor
    """
    exp = matrix_exp(w * w)
    return exp.trace() - w.shape[0]


def sample_gumbel(shape, eps=1e-20, seed=0):
    eps = Tensor(eps)
    u = ops.uniform(shape, minval=Tensor(0.), maxval=Tensor(1.), seed=seed)
    u = -ops.log(-ops.log(u + eps) + eps)
    u[list(range(shape[0])), list(range(shape[0]))] = 0
    return u


def gumbel_sigmoid(logits, temperature, seed):
    gumbel_softmax_sample = (logits
                             + sample_gumbel(logits.shape, seed=seed)
                             - sample_gumbel(logits.shape, seed=seed + 1))
    y = nn.Sigmoid()(gumbel_softmax_sample / temperature)
    return y


def generate_upper_triangle_indices(d):
    if d == 1:
        return ops.Zeros()((1,), mstype.int64)
    mat = np.arange(d ** 2).reshape(d, d)
    target_indices = mat[np.triu(mat, 1) != 0]
    return Tensor.from_numpy(target_indices)


def callback_after_training(w_logits, temperature, graph_thresh):
    d = w_logits.shape[0]
    w_final_weight = convert_logits_to_sigmoid(w_logits, temperature)
    w_final = w_final_weight.copy()  # final graph
    w_final[w_final <= graph_thresh] = 0  # Thresholding
    w_final[w_final > graph_thresh] = 1
    w_final[list(range(d)), list(range(d))] = 0  # Mask diagonal to 0
    return w_final, w_final_weight


def nonzero(input):
    """
    Return indexes of all nonzero/True elements.
    Parameters
    ----------
    input

    Returns
    -------

    """
    output = msnp.zeros((1, input.shape[0]), dtype=mstype.float32)
    for r, rows in enumerate(input):
        output[:, r] = Tensor([-1])
        if rows != 0:
            output[:, r] = Tensor([r])
    return output[0]


def matrix_exp(matrix):
    """
    compute matrix exponential

    Parameters
    ----------
    matrix: ms.Tensor

    Returns
    -------
    expm: matrix exponential value of A
    """
    expm_val = msnp.zeros(matrix.shape, dtype=mstype.float32)
    eye_mat = msnp.eye(matrix.shape[0], dtype=mstype.float32)
    k = Tensor(1.0, mstype.float32)

    while msnp.norm(eye_mat, 1) > 0:
        expm_val = expm_val + eye_mat
        eye_mat = msnp.matmul(matrix, eye_mat) / k
        k += 1.0
    return expm_val
