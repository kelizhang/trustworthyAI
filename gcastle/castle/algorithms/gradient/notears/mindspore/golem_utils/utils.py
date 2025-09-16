# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Ignavier Ng (https://github.com/ignavier/golem)
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

import mindspore
import numpy as np
import networkx as nx
from mindspore import numpy as msnp


def is_dag(B):
    """
    Check whether B corresponds to a DAG.

    Parameters
    ----------
    B: numpy.ndarray
        [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def matrix_exp(matrix, float_type):
    """
    compute matrix exponential

    Parameters
    ----------
    matrix: ms.Tensor

    Returns
    -------
    expm: matrix exponential value of A
    """
    expm_val = msnp.zeros(matrix.shape, dtype=float_type)
    eye_mat = msnp.eye(matrix.shape[0], dtype=float_type)
    k = mindspore.Tensor(1.0, float_type)

    while msnp.norm(eye_mat, 1) > 0:
        expm_val = expm_val + eye_mat
        eye_mat = msnp.matmul(matrix, eye_mat) / k
        k += 1.0

    return expm_val


def slogdet(a):
    """

    Parameters
    ----------
    a

    Returns
    -------

    """
    sign, logdet = np.linalg.slogdet(a.asnumpy())
    return mindspore.Tensor(sign), mindspore.Tensor(logdet)
