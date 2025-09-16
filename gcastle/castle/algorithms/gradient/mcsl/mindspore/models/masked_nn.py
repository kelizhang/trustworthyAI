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

import mindspore
import mindspore.nn as nn


class MaskedNN(nn.Cell):

    def __init__(self, mask, num_hidden_layers, hidden_dim) -> None:
        super(MaskedNN, self).__init__()
        self.mask = mask  # use mask to determine input dimension
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

        self._init_nn()

    def construct(self, x, choice) -> mindspore.Tensor:
        """

        Parameters
        ----------
        x: mindspore.Tensor
            possible parents
        choice: str of int
            current sub-note y

        Returns
        -------
        output: mindspore.Tensor
            shape = (n,)
        """
        output = self.nets[choice](x)

        return output

    def _init_nn(self):
        """ Initialize net for each node"""

        md = list()
        for i in range(self.mask.shape[0]):
            reg_nn = []
            for j in range(self.num_hidden_layers):
                input_dim = self.mask.shape[1] if j == 0 else self.hidden_dim
                reg_nn.extend([nn.Dense(in_channels=input_dim,
                                        out_channels=self.hidden_dim,
                                        weight_init="xavier_uniform"),
                               nn.LeakyReLU(alpha=0.05)])
            reg_nn.append(nn.Dense(in_channels=self.hidden_dim,
                                   out_channels=1,
                                   weight_init="xavier_uniform"))
            reg_nn = nn.SequentialCell(*reg_nn)
            md.append(reg_nn)

        self.nets = nn.CellList(md)
