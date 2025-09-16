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
import mindspore.nn as nn

from .utils import matrix_exp, slogdet


class GolemModel(nn.Cell):
    """
    Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances,
                 B_init=None, float_type=None):
        """
        Initialize self.

        Parameters
        ----------
        n: int
            Number of samples.
        d: int
            Number of nodes.
        lambda_1: float
            Coefficient of L1 penalty.
        lambda_2: float
            Coefficient of DAG penalty.
        equal_variances: bool
            Whether to assume equal noise variances
            for likelibood objective. Default: True.
        B_init: numpy.ndarray or None
            [d, d] weighted matrix for initialization. 
            Set to None to disable. Default: None.
        float_type: mindspore.dtype
            the data‘s type.
        """

        super().__init__()
        self.n = n
        self.d = d
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init
        self.float_type = float_type

        if self.B_init is not None:
            self._B = mindspore.Parameter(mindspore.tensor(self.B_init))
        else:
            self._B = mindspore.Parameter(mindspore.ops.zeros((self.d, self.d), self.float_type))

    def construct(self, X):
        """Build tensorflow graph."""
        # Placeholders and variables
        B = self.preprocess()

        # Likelihood, penalty terms and score
        likelihood = self._compute_likelihood(B, X)
        l1_penalty = self._compute_L1_penalty(B)
        h = self._compute_h(B)
        score = (likelihood + self.lambda_1 * l1_penalty + self.lambda_2 * h)
        return score

    def preprocess(self):
        """
        Set the diagonals of B to zero.

        Parameters
        ----------
        B: tf.Tensor
            [d, d] weighted matrix.

        Return
        ------
        torch.Tensor: [d, d] weighted matrix.
        """

        return (mindspore.ops.Ones()((self.d, self.d), self.float_type)
                - mindspore.ops.eye(self.d, self.d, self.float_type)) * self._B

    def _compute_likelihood(self, B, X):
        """
        Compute (negative log) likelihood in the linear Gaussian case.

        Return
        ------
        mindspore.Tensor: Likelihood term (scalar-valued).
        """

        diagonal_matrix = mindspore.ops.eye(self.d, self.d, self.float_type) - B
        if mindspore.__version__ < "2.0":
            slog_det = slogdet(diagonal_matrix)[1]
        else:
            slog_det = mindspore.ops.slogdet(diagonal_matrix)[1]

        if self.equal_variances:  # Assuming equal noise variances
            return (0.5 * self.d
                    * mindspore.ops.log(mindspore.ops.Square()(
                        mindspore.numpy.norm(X - mindspore.ops.matmul(X, B), axis=[0, 1])))
                    - slog_det)
        else:  # Assuming non-equal noise variances
            return (0.5
                    * mindspore.ops.ReduceSum()(mindspore.ops.log(
                        mindspore.ops.ReduceSum()(mindspore.ops.square(X - mindspore.ops.matmul(X, B)))))
                    - slog_det)

    def _compute_L1_penalty(self, B):
        """
        Compute L1 penalty.

        Return
        ------
        tf.Tensor: L1 penalty term (scalar-valued).
        """
        return mindspore.numpy.norm(B, axis=[0, 1], ord=1)

    def _compute_h(self, B):
        """
        Compute DAG penalty.

        Return
        ------
        mindspore.Tensor: DAG penalty term (scalar-valued).
        """

        return matrix_exp(B * B, self.float_type).trace() - self.d
