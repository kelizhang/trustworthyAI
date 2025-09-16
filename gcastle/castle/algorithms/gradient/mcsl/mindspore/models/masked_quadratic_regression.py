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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from ..helpers.utils import generate_upper_triangle_indices


class MaskedQuadraticRegression(nn.Cell):

    def __init__(self, mask, n_samples, n_nodes, seed) -> None:
        super(MaskedQuadraticRegression, self).__init__()
        self.mask = mask
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.seed = seed

        self.target_indices = generate_upper_triangle_indices(n_nodes)
        w3 = mindspore.Tensor(np.random.uniform(low=-0.05, high=0.05, size=(len(self.target_indices),)),
                              dtype=mindspore.float32)
        self.w3 = mindspore.Parameter(w3)
        self._init_weight()

    def construct(self, x, choice) -> mindspore.Tensor:
        """

        Parameters
        ----------
        x: mindspore.Tensor
            shape = (n, d - 1)
        choice

        Returns
        -------
        output: mindspore.Tensor
            a vector of shape = (n,)
        """

        output = mindspore.ops.Zeros()((self.n_samples,), mindspore.float32)

        # Linear terms
        output += (self.weight[choice * 2 + 0] * x).sum(axis=1)

        # Squared terms
        output += (self.weight[choice * 2 + 1] * mindspore.ops.Square()(x)).sum(axis=1)

        # Cross terms
        x_ = ops.ExpandDims()(x, 1)
        y_ = ops.ExpandDims()(x, 2)

        all_cross_terms = mindspore.ops.Reshape()(x_ * y_, (self.n_samples, -1))
        combinations_cross_terms = all_cross_terms[:, self.target_indices]

        output += (self.w3 * combinations_cross_terms).sum(axis=1)
        return output

    def _init_weight(self):
        md = list()
        for i in range(self.mask.shape[0]):
            w1 = mindspore.Tensor(np.random.uniform(
                low=-0.05, high=0.05, size=(self.mask.shape[1],)), mindspore.float32)
            self.w1 = mindspore.Parameter(w1, name="w1"+str(i))

            w2 = mindspore.Tensor(np.random.uniform(
                low=-0.05, high=0.05, size=(self.mask.shape[1],)), mindspore.float32)
            self.w2 = mindspore.Parameter(w2, name="w2"+str(i))
            md.extend([self.w1, self.w2])

        self.weight = mindspore.ParameterTuple(tuple(md))
