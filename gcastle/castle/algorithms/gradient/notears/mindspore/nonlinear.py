# coding=utf-8
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Xun Zheng (https://github.com/xunzheng/notears)
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
from mindspore import nn

from castle.common import GradientBasedLearner, Tensor
from castle.common.consts import NONLINEAR_NOTEARS_VALID_PARAMS
from castle.common.validator import check_args_value
from castle.common.class_factory import ClassFactory

from .models import MLPModel, SobolevModel, NotearsWithLossCell, L1Loss, GradWrap
from .utils.lbfgsb_scipy import LBFGSBScipy


@ClassFactory.algorithms.register()
class NotearsNonlinear(GradientBasedLearner):
    """
    Notears Nonlinear.
    include notears-mlp and notears-sob.
    A gradient-based algorithm using neural network or Sobolev space modeling for non-linear causal relationships.

    Parameters
    ----------
    lambda1: float 
        l1 penalty parameter
    lambda2: float 
        l2 penalty parameter
    iterations: int
        max num of dual ascent steps
    h_tol: float 
        exit if |h(w_est)| <= htol
    rho_max: float 
        exit if rho >= rho_max
    w_threshold: float 
        drop edge if |weight| < threshold
    hidden_layers: Iterrable
        Dimension of per hidden layer, and the last element must be 1 as output dimension.
        At least contains 2 elements. For example: hidden_layers=(5, 10, 1), denotes two hidden
        layer has 5 and 10 dimension and output layer has 1 dimension.
        It is effective when model_type='mlp'.
    expansions: int
        expansions of each variable, it is effective when model_type='sob'.
    bias: bool
        Indicates whether to use weight deviation.
    model_type: str
        The Choice of Two Nonlinear Network Models in a Notears Framework:
        Multilayer perceptrons value is 'mlp', Basis expansions value is 'sob'.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1909.13189
    
    Examples
    --------
    >>> import castle
    >>> castle.set_backend('mindspore')
    >>> from castle.algorithms import NotearsNonlinear
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = NotearsNonlinear()
    # If you want to use GPU, you can call CORL().set_context() like the following code:
    >>> algo.set_context(device_target='GPU')  # 'GPU', 'CPU' or 'Ascend'
    # Or if your backend is mindspore, and you want to use ms.PYNATIVE_MODE, you can:
    >>> import mindspore as ms
    >>> algo.set_context(device_target='GPU', mode=ms.PYNATIVE_MODE)
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    @check_args_value(NONLINEAR_NOTEARS_VALID_PARAMS)
    def __init__(self, lambda1: float = 0.01,
                 lambda2: float = 0.01,
                 iterations: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 hidden_layers: tuple = (10, 1),
                 expansions: int = 10,
                 bias: bool = True,
                 model_type: str = "mlp",
                 learning_rate: float = 1e-3):

        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.iterations = iterations
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.hidden_layers = hidden_layers
        self.expansions = expansions
        self.bias = bias
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.rho, self.alpha, self.h = 1.0, 0.0, np.inf


    def learn(self, data, columns=None, **kwargs):
        """
        Set up and run the NotearsNonlinear algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """

        data = data.astype(np.float32)
        X = Tensor(data, columns=columns)

        input_dim = X.shape[1]
        model = self.get_model(input_dim)
        W_est = self.notears_nonlinear(model, X.data, self.learning_rate)
        causal_matrix = (abs(W_est) > self.w_threshold).astype(int)
        self.weight_causal_matrix = Tensor(W_est,
                                           index=X.columns,
                                           columns=X.columns)
        self.causal_matrix = Tensor(causal_matrix, index=X.columns, columns=X.columns)

    def dual_ascent_step(self, model, X, learning_rate):
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Parameters
        ----------
        model: nn.Module
            network model
        X: torch.tenser
            sample data
        learning_rate: float
            optimizer learning rate
        Returns
        -------
        :tuple
            cycle control parameter
        """
        h_new = None
        X_mindspore = mindspore.Tensor(X)
        # Setting the Loss Function
        net_with_criterion = NotearsWithLossCell(model, L1Loss())
        # Set Optimizer
        optimizer = LBFGSBScipy(net_with_criterion, params=model.trainable_params(), learning_rate=learning_rate)

        train_network = GradWrap(net_with_criterion)
        while self.rho < self.rho_max:
            def closure():
                p_dict = dict()
                p_dict["rho"] = self.rho
                p_dict["alpha"] = self.alpha
                p_dict["lambda2"] = self.lambda2
                p_dict["lambda1"] = self.lambda1
                loss = net_with_criterion(X_mindspore, X_mindspore, p_dict)
                # Find the Gradient
                grads = train_network(X_mindspore, X_mindspore, p_dict)
                return loss, grads

            optimizer.step(closure)  # NOTE: updates model in-place
            h_new = model.h_func()
            h_new = h_new.asnumpy()
            if h_new > 0.25 * self.h:
                self.rho *= 10
            else:
                break
        self.alpha += self.rho * h_new
        self.h = h_new

    def notears_nonlinear(self, model: nn.Cell, X: np.ndarray, learning_rate: float):
        """
        notaears frame entrance.

        Parameters
        ----------
        model: nn.Module
            network model
        X: castle.Tensor or numpy.ndarray
            sample data
        learning_rate: float
            optimizer learning rate
        Returns
        -------
        :tuple
            Prediction Graph Matrix Coefficients.
        """

        logging.info('[start]: n={}, d={}, iter_={}, h_={}, rho_={}'.format(
            X.shape[0], X.shape[1], self.iterations, self.h_tol, self.rho_max))

        for _ in range(self.iterations):
            self.dual_ascent_step(model, X, learning_rate)

            logging.debug('[iter {}] h={:.3e}, rho={:.1e}'.format(_, self.h, self.rho))

            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break
        W_est = model.fc1_to_adj()

        logging.info('FINISHED')

        return W_est

    def get_model(self, input_dim):
        """
            Choose a different model.
        Parameters
        ----------
        input_dim: int
            Enter the number of data dimensions.

        Returns
        -------

        """
        if self.model_type == "mlp":
            model = MLPModel(dims=[input_dim, *self.hidden_layers], bias=self.bias)
            return model
        elif self.model_type == "sob":
            model = SobolevModel(input_dim, k=self.expansions, bias=self.bias)
            return model
        else:
            raise ValueError(f'Unsupported model type {self.model_type}.')
