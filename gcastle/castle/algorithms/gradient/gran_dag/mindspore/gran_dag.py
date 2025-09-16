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
from scipy.linalg import expm
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as msnp

from .base import NonlinearGauss
from .base import NonlinearGaussANM
from .base import compute_jacobian_avg
from .base import is_acyclic
from .base import GranLoss

from castle.common import GradientBasedLearner, Tensor
from castle.common.validator import check_args_value
from castle.common.consts import GRANDAG_VALID_PARAMS
from castle.common.class_factory import ClassFactory


class NormalizationData(object):
    """
    Create Normalization Data object

    Parameters
    ----------
    data: numpy.ndarray
        train x
    normalize: bool, default False
        whether normalization
    mean: float or None, default None
        Mean value of normalization
    std: float or None, default None
        Standard Deviation of normalization
    shuffle: bool, default False
        whether shuffle
    train_size: float, default 0.8
        ratio of train data for training
    train: bool, default True
        whether training
    random_seed: int, default 42
        for set random seed
    """

    def __init__(self, data, normalize=False, mean=None, std=None,
                 shuffle=False, train_size=0.8, train=True, random_seed=42):
        self.random = np.random.RandomState(random_seed)

        shuffle_idx = np.arange(data.shape[0])
        if shuffle:
            self.random.shuffle(shuffle_idx)

        if isinstance(train_size, float):
            train_samples = int(data.shape[0] * train_size)
        else:
            raise TypeError("The param train_size must be float < 1")
        if train:
            data = data[shuffle_idx[: train_samples]]
        else:
            data = data[shuffle_idx[train_samples:]]
        # as tensor
        self.data_set = ms.Tensor(data).astype(dtype=ms.float32)

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if mean is None or std is None:
                self.mean = msnp.mean(self.data_set, 0, keepdims=True)
                self.std = msnp.std(self.data_set, 0, keepdims=True)
            self.data_set = (self.data_set - self.mean) / self.std
        self.n_samples = self.data_set.shape[0]

    def sample(self, batch_size):
        """
        sampling from self.dataset

        Parameters
        ----------
        batch_size: int
            batch size of sample

        Returns
        -------
        samples: mindspore.Tensor
            sample data after sampling
        ops.ones_like(samples): mindspore.Tensor
        """
        sample_idxs = self.random.choice(np.arange(int(self.n_samples)),
                                         size=(int(batch_size),),
                                         replace=False)
        samples = self.data_set[ms.Tensor(sample_idxs, ms.int32)]

        return samples, ops.ones_like(samples)


@ClassFactory.algorithms.register()
class GraNDAG(GradientBasedLearner):
    """
    Gradient Based Neural DAG Learner

    A gradient-based algorithm using neural network modeling for
    non-linear additive noise data

    References: https://arxiv.org/pdf/1906.02226.pdf

    Parameters
    ----------
    hidden_num: int, default 2
        number of hidden layers
    hidden_dim: int, default 10
        number of dimension per hidden layer
    batch_size: int, default 64
        batch size of per training of NN
    lr: float, default 0.001
        learning rate
    lr_decay: int, default: 1000
        Period of learning rate decay.
    gamma: float, default: 0.1
        Multiplicative factor of learning rate decay.
    iterations: int, default 10000
        times of iteration
    model_name: str, default 'NonLinGaussANM'
        model name, 'NonLinGauss' or 'NonLinGaussANM'
    nonlinear: str, default 'leaky-relu'
        name of Nonlinear activation function, 'sigmoid' or 'leaky-relu'
    optimizer: str, default 'rmsprop'
        Method of optimize, 'rmsprop' or 'sgd'
    h_threshold: float, default 1e-8
        constrained threshold, if constrained value less than equal h_threshold
        means augmented lagrangian has converged, model will stop trainning
    use_pns: bool, default False
        whether use pns before training, if nodes > 50, use it.
    pns_thresh: float, default 0.75
        threshold for feature importance score in pns
    num_neighbors: int, default None
        number of potential parents for each variables
    normalize: bool, default False
        whether normalize data
    random_seed: int, default 42
        random seed
    norm_prod: str, default 'paths'
        use norm of product of paths, 'none' or 'paths'
        'paths': use norm, 'none': with no norm
    square_prod: bool, default False
        use squared product of paths
    jac_thresh: bool, default True
        get the average Jacobian with the trained model
    lambda_init: float, default 0.0
        initialization of Lagrangian coefficient in the optimization of
        augmented Lagrangian
    mu_init: float, default 0.001
        initialization of penalty coefficient in the optimization of
        augmented Lagrangian
    omega_lambda: float, default 0.0001
        tolerance on the delta lambda, to find saddle points
    omega_mu: float, default 0.9
        check whether the constraint decreases sufficiently if it decreases
        at least (1-omega_mu) * h_prev
    stop_crit_win: int, default 100
        number of iterations for updating values
    edge_clamp_range: float, default 0.0001
        threshold for keeping the edge (if during training)

    Examples
    --------
        Load data
    >>> from castle.datasets import load_dataset
    >>> data, true_dag, _ = load_dataset('IID_Test')

    >>> gnd = GraNDAG(input_dim=data.shape[1])
    >>> gnd.learn(data=data)

        Also print GraN_DAG.model.adjacency with mindspore.Tensor type
        or print GranN_DAG.causal_matrix with numpy.ndarray.
    >>> print(gnd.causal_matrix)
    >>> print(gnd.model.adjacency)
    """

    OUT_DIM = 2
    OUT_DIM_ANM = 1

    @check_args_value(GRANDAG_VALID_PARAMS)
    def __init__(self,
                 hidden_num=2,
                 hidden_dim=10,
                 batch_size=64,
                 lr=0.01,
                 lr_decay=1000,
                 gamma=0.1,
                 iterations=10000,
                 model_name='NonLinGaussANM',
                 nonlinear='leaky-relu',
                 optimizer='rmsprop',
                 h_threshold=1e-8,
                 use_pns=False,
                 pns_thresh=0.75,
                 num_neighbors=None,
                 normalize=False,
                 random_seed=42,
                 jac_thresh=True,
                 lambda_init=0.0,
                 mu_init=0.001,
                 omega_lambda=0.0001,
                 omega_mu=0.9,
                 stop_crit_win=100,
                 edge_clamp_range=0.0001,
                 norm_prod='paths',
                 square_prod=False
                 ):
        super(GraNDAG, self).__init__()

        self.input_dim = None
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.iterations = iterations
        self.model_name = model_name
        self.nonlinear = nonlinear
        self.optimizer = optimizer
        self.h_threshold = h_threshold
        self.use_pns = use_pns
        self.pns_thresh = pns_thresh
        self.num_neighbors = num_neighbors
        self.normalize = normalize
        self.random_seed = random_seed
        self.jac_thresh = jac_thresh
        self.lamb = lambda_init
        self.mu = mu_init
        self.omega_lambda = omega_lambda
        self.omega_mu = omega_mu
        self.stop_crit_win = stop_crit_win
        self.edge_clamp_range = edge_clamp_range
        self.norm_prod = norm_prod
        self.square_prod = square_prod

        if self.norm_prod not in ['paths', 'none']:
            raise ValueError("Param 'norm_prod' should be in {'paths', 'none'}.")

    def learn(self, data, columns=None, **kwargs):
        """Set up and run the Gran-DAG algorithm

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            include Tensor.data
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """

        # Control as much randomness as possible
        self._set_seed(self.random_seed)

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        self.input_dim = data.shape[1]

        if data.shape[1] != self.input_dim:
            raise ValueError("The number of variables is `{}`, "
                             "the param input_dim is `{}`, "
                             "they must be consistent.".format(data.shape[1],
                                                               self.input_dim))

        if self.model_name == "NonLinGauss":
            self.model = NonlinearGauss(input_dim=self.input_dim,
                                        hidden_num=self.hidden_num,
                                        hidden_dim=self.hidden_dim,
                                        output_dim=self.OUT_DIM,
                                        nonlinear=self.nonlinear)
        elif self.model_name == "NonLinGaussANM":
            self.model = NonlinearGaussANM(input_dim=self.input_dim,
                                           hidden_num=self.hidden_num,
                                           hidden_dim=self.hidden_dim,
                                           output_dim=self.OUT_DIM_ANM,
                                           nonlinear=self.nonlinear)
        else:
            raise ValueError(
                "model has to be in {NonLinGauss, NonLinGaussANM}")

        # create NormalizationData
        train_data = NormalizationData(data.data, train=True,
                                       normalize=self.normalize)
        test_data = NormalizationData(data.data, train=False,
                                      normalize=self.normalize,
                                      mean=train_data.mean,
                                      std=train_data.std)

        # apply preliminary neighborhood selection if input_dim > 50
        if self.use_pns:
            if self.num_neighbors is None:
                num_neighbors = self.input_dim
            else:
                num_neighbors = self.num_neighbors

            self.model = neighbors_selection(model=self.model,
                                             all_samples=data.data,
                                             num_neighbors=num_neighbors,
                                             thresh=self.pns_thresh)

        # update self.model by train
        self._train(train_data=train_data, test_data=test_data)

        # get DAG by run _to_dag
        self._to_dag(train_data)

        self._causal_matrix = Tensor(self.model.adjacency.asnumpy(),
                                     index=data.columns, columns=data.columns)

    def _train(self, train_data, test_data):
        """
        Applying augmented Lagrangian to solve the continuous constrained problem.

        Parameters
        ----------
        train_data: NormalizationData object
            train samples
        test_data: NormalizationData object
            test samples for validation
        """

        # Initialize stuff for learning loop
        aug_lagrangians_val = []

        hs = []
        not_nlls = []  # Augmented Lagrangian minus (pseudo) NLL

        lr_strategy = nn.ExponentialDecayLR(learning_rate=self.lr,
                                            decay_rate=self.gamma,
                                            decay_steps=self.lr_decay)
        if self.optimizer == "sgd":
            optimizer = nn.optim.SGD(self.model.trainable_params(),
                                     learning_rate=lr_strategy)
        elif self.optimizer == "rmsprop":
            optimizer = nn.optim.RMSProp(self.model.trainable_params(),
                                         learning_rate=lr_strategy)
        else:
            raise ValueError("optimizer should be in {'sgd', 'rmsprop'}")

        # Package training information
        run_mode = 'pynative'
        if ms.get_context('mode') == ms.GRAPH_MODE:
            run_mode = 'graph'
        self.loss_fn = GranLoss(input_dim=self.input_dim,
                                mu=self.mu, lamb=self.lamb,
                                norm_prod=self.norm_prod,
                                square_prod=self.square_prod,
                                run_mode=run_mode)
        net_loss = nn.WithLossCell(self.model, self.loss_fn)
        train_cell = nn.TrainOneStepCell(net_loss, optimizer)

        y_label = ms.Tensor(np.array([0]*self.batch_size, dtype=np.float32))
        # Learning loop:
        for iter_num in tqdm(range(self.iterations), desc='Training Iterations'):
            x, _ = train_data.sample(self.batch_size)

            weights = self.model.get_weight_bias_parameters(mode=('w', ))[0]
            w_adj = self.loss_fn.get_w_adj(weights)
            expm_input = expm(w_adj.asnumpy())
            h = np.trace(expm_input) - self.input_dim

            # model train
            self.model.set_train(True)
            train_cell(x, y_label)
            self.model.set_train(False)

            # clamp edges, thresholding
            if self.edge_clamp_range != 0:
                to_keep = (w_adj > self.edge_clamp_range) * 1
                self.loss_fn.adjacency *= to_keep
                self.model.adjacency *= to_keep

            not_nlls.append(0.5 * self.loss_fn.mu * h ** 2
                            + self.loss_fn.lamb * h)

            # compute loss on whole validation set
            if iter_num % self.stop_crit_win == 0:
                x, _ = test_data.sample(test_data.n_samples)
                loss_val = - msnp.mean(self.model.compute_log_likelihood(x))
                aug_lagrangians_val.append(
                    [iter_num, loss_val.asnumpy().item() + not_nlls[-1]])

            # compute delta for lambda
            if iter_num >= 2 * self.stop_crit_win \
                    and iter_num % (2 * self.stop_crit_win) == 0:
                t0 = aug_lagrangians_val[-3][1]
                t_half = aug_lagrangians_val[-2][1]
                t1 = aug_lagrangians_val[-1][1]

                # if the validation loss went up and down,
                # do not update lagrangian and penalty coefficients.
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0) / self.stop_crit_win
            else:
                delta_lambda = -np.inf  # do not update lambda nor mu

            # Does the augmented lagrangian converged?
            # if h value less than equal self.h_threshold value,
            # means augmented lagrangian has converged, stop model training
            if h > self.h_threshold:
                # if we have found a stationary point of the augmented loss
                if abs(delta_lambda) < self.omega_lambda or delta_lambda > 0:
                    self.loss_fn.lamb += self.loss_fn.mu * h

                    # Did the constraint improve sufficiently?
                    hs.append(h)
                    if len(hs) >= 2:
                        if hs[-1] > hs[-2] * self.omega_mu:
                            self.loss_fn.mu *= 10

                    # little hack to make sure the moving average is going down.
                    gap_in_not_nll = 0.5 * self.loss_fn.mu * h ** 2 +\
                                     self.loss_fn.lamb * h - not_nlls[-1]
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                    if self.optimizer == "rmsprop":
                        optimizer = nn.optim.RMSProp(self.model.trainable_params(),
                                                     learning_rate=lr_strategy)
                    else:
                        optimizer = nn.optim.SGD(self.model.trainable_params(),
                                                 learning_rate=lr_strategy)
                    net_loss = nn.WithLossCell(self.model, self.loss_fn)
                    train_cell = nn.TrainOneStepCell(net_loss, optimizer)
            else:
                # Final clamping of all edges == 0
                to_keep = (w_adj > 0).astype(ms.float32)
                self.loss_fn.adjacency *= to_keep
                self.model.adjacency *= to_keep

                return self.model

    def _to_dag(self, train_data):
        """
        threshold: from weaker to stronger in A
        1- If some entries of A less than equal threshold, mask them to zero
        (This can happen with stochastic proximal gradient descent)
        2- Remove edges by threshold until a DAG is obtained.

        Parameters
        ----------
        train_data: NormalizationData
            train samples
        """

        if self.jac_thresh:
            jacobian_adj = compute_jacobian_avg(
                self.model, train_data, train_data.n_samples).transpose()
        else:
            weights = self.model.get_weight_bias_parameters(mode=('w', ))[0]
            jacobian_adj = self.loss_fn.get_w_adj(weights)

        dia_index = msnp.diag_indices(jacobian_adj.shape[0])
        jacobian_adj[dia_index] = 0

        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(jacobian_adj.asnumpy())
        epsilon = 1e-8
        for step, t in enumerate(thresholds):
            to_keep = (jacobian_adj > t + epsilon).astype(dtype=ms.float32)
            new_adj = self.model.adjacency * to_keep
            if is_acyclic(new_adj):
                self.model.adjacency = new_adj
                break

        return self.model


def neighbors_selection(model, all_samples, num_neighbors, thresh):
    """
    Preliminary neighborhood selection
    After pns, just model.adjacency is changed. if nodes > 50, use it.

    Parameters
    ----------
    model: model object
    all_samples: array-like
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    out: model
    """

    model_adj = model.adjacency.asnumpy()
    model_adj = _pns(model_adj, all_samples, num_neighbors, thresh)
    model.adjacency = ms.Tensor(model_adj).copy()

    return model


def _pns(model_adj, all_samples, num_neighbors, thresh):
    """
    Preliminary neighborhood selection

    Parameters
    ----------
    model_adj: numpy.ndarray
        adjacency matrix, all element is 1
    all_samples: numpy.ndarray
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    model_adj: numpy.ndarray
        adjacency matrix, after pns process
    """

    num_nodes = all_samples.shape[1]

    for node in tqdm(range(num_nodes), desc='Preliminary neighborhood selection'):
        x_other = np.copy(all_samples)
        x_other[:, node] = 0
        extra_tree = ExtraTreesRegressor(n_estimators=500)
        extra_tree.fit(x_other, all_samples[:, node])
        selected_reg = SelectFromModel(extra_tree,
                                       threshold="{}*mean".format(thresh),
                                       prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False)
        model_adj[:, node] *= mask_selected

    return model_adj
