# coding = utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import math
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
import mindspore.numpy as msnp


def get_dataloader(data, batch_size=100):
    """Get data loader"""
    def get_data(x):
        for i in range(x.shape[0]):
            yield x[i, :]

    dataloader = ds.GeneratorDataset(list(get_data(data)),
                                     column_names=["x"],
                                     shuffle=True)
    dataloader = dataloader.batch(batch_size=batch_size)

    return dataloader


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = msnp.ones((num_nodes, num_nodes))
    eye = msnp.eye(num_nodes, num_nodes)
    triu_indices = (msnp.triu(ones) - eye).nonzero().transpose()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = msnp.ones((num_nodes, num_nodes))
    eye = msnp.eye(num_nodes, num_nodes)
    tril_indices = (msnp.tril(ones) - eye).nonzero().transpose()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = msnp.ones((num_nodes, num_nodes))
    eye = msnp.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().transpose()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = msnp.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = msnp.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def matrix_poly(matrix, d):
    x = msnp.eye(d, dtype=ms.float64) + ops.div(matrix, d)
    return msnp.matrix_power(x, d)


def _h_A(A, m):
    """compute constraint h(A) value"""
    expm_A = matrix_poly(A*A, m)
    h_A = msnp.trace(expm_A, dtype=ms.float64) - m
    return h_A


def compute_lr(original_lr, c_A):
    """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        learning_rate = MAX_LR
    elif estimated_lr < MIN_LR:
        learning_rate = MIN_LR
    else:
        learning_rate = estimated_lr

    return learning_rate


def nll_gaussian(preds, target, variance=0.0, add_const=False):
    """calculate reconstruction accuracy loss"""
    mean1 = preds
    mean2 = target
    neg_log_p = variance + ops.div(ops.pow(mean1 - mean2, 2),
                                   2.*np.exp(2. * variance))
    if add_const:
        const = 0.5 * ops.log(2 * ms.Tensor(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / target.shape[0]


def kl_gaussian_sem(preds):
    """KL-divergence term loss"""
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / preds.shape[0])*0.5


def a_connect_loss(A, tol, z):
    """matrix loss: makes sure at least A connected to another parents for child"""
    d = A.shape[0]
    loss = 0
    for i in range(d):
        loss += 2 * tol - msnp.sum(msnp.abs(A[:, i])) - msnp.sum(
            msnp.abs(A[i, :])) + z * z
    return loss


def a_positive_loss(A, z_positive):
    """element loss: make sure each A_ij > 0"""
    result = - A + z_positive * z_positive
    loss = msnp.sum(result)

    return loss


def get_optimizer(optimizer_name, encoder, decoder, lr_strategy):
    """get optimizer"""
    if optimizer_name.lower() == 'adam':
        optimizer = nn.Adam(params=[{'params': encoder.trainable_params()},
                                    {'params': decoder.trainable_params()}],
                            learning_rate=lr_strategy)
    elif optimizer_name.lower() == 'sgd':
        optimizer = nn.SGD(params=[{'params': encoder.trainable_params()},
                                   {'params': decoder.trainable_params()}],
                           learning_rate=lr_strategy)
    else:
        raise ValueError("optimizer should be in {'Adam', 'SGD'}")

    return optimizer
