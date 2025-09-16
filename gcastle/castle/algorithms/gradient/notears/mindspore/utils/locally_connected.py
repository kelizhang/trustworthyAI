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
import math
import numpy as np

from mindspore import Parameter, ops, Tensor, nn
from mindspore.common.initializer import Uniform


class LocallyConnected(nn.Cell):
    """
    Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Parameters
    ----------
    num_linear: num of local linear layers, i.e.
    input_features: m1
        Shape: [n, d, m1]
    output_features: m2
        Shape: [n, d, m2]
    bias: whether to include bias or not

    Attributes
    ----------
    weight: [d, m1, m2]
    bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.w = np.zeros([num_linear,
                           input_features,
                           output_features]).astype(np.float32)
        self.is_bias = bias
        self.weight = None
        self.bias = None
        if bias:
            self.b = np.zeros([num_linear,
                               output_features]).astype(np.float32)
        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        uniform = Uniform(scale=bound)
        uniform(self.w)
        self.weight = Parameter(Tensor(self.w), name="weight")
        if self.is_bias:
            uniform(self.b)
            self.bias = Parameter(Tensor(self.b), name="bias")

    def construct(self, input_x: Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = ops.matmul(ops.ExpandDims()(input_x, 2),
                         ops.ExpandDims()(self.weight, 0)) 
        out = out.squeeze(axis=2)
        if self.is_bias:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        """
        (Optional)Set the extra information about this module. You can test
        it by printing an object of this class.

        Returns
        -------

        """

        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
