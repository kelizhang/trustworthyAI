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

import logging
from tqdm import tqdm
import platform
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from .data_loader import DataGenerator_read_data
from .models import Actor
from .rewards import GetReward
from .helpers.lambda_utils import BIC_lambdas
from .helpers.analyze_utils import convert_graph_int_to_adj_mat
from .helpers.analyze_utils import graph_prunned_by_coef, graph_prunned_by_coef_2nd

from castle.common import GradientBasedLearner, Tensor
from castle.metrics import StructureMetrics
from castle.common.consts import RL_VALID_PARAMS
from castle.common.validator import check_args_value
from castle.common.class_factory import ClassFactory


class NetWithLossCell(nn.Cell):
    """
    Packaging network and loss

    Parameters
    ----------
    actor: Actor network instantiate object
    call_reward: Decoder network instantiate object
    lambda1:
    lambda2:
    """
    def __init__(self, actor, call_reward, lambda1, lambda2):
        super(NetWithLossCell, self).__init__()
        self.actor = actor
        self.call_reward = call_reward
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def construct(self, input_x):
        # actor
        graphs_feed, log_softmax, entropy_regularization, critic_predictions = self.actor.build_permutation(input_x)
        reward_feed = self.call_reward.cal_rewards(graphs_feed.asnumpy(), self.lambda1, self.lambda2)  # np.array
        loss = self.actor.build_reward(reward=-reward_feed[:, 0].astype(ms.float32),
                                       log_softmax=log_softmax,
                                       entropy_regularization=entropy_regularization,
                                       critic_predictions=critic_predictions)

        return loss


ClassFactory.algorithms.register()
class RL(GradientBasedLearner):
    """
    RL Algorithm.
    A RL-based algorithm that can work with flexible score functions (including non-smooth ones).

    Parameters
    ----------
    encoder_type: str
        type of encoder used
    hidden_dim: int
        actor LSTM num_neurons
    num_heads: int
        actor input embedding
    num_stacks: int
        actor LSTM num_neurons
    residual: bool
        whether to use residual for gat encoder
    decoder_type: str
        type of decoder used
    decoder_activation: str
        activation for decoder
    decoder_hidden_dim: int
        hidden dimension for decoder
    use_bias: bool
        Whether to add bias term when calculating decoder logits
    use_bias_constant: bool
        Whether to add bias term as CONSTANT when calculating decoder logits
    bias_initial_value: float
        Initial value for bias term when calculating decoder logits
    batch_size: int
        batch size for training
    input_dimension: int
        dimension of reshaped vector
    normalize: bool
        whether the inputdata shall be normalized
    transpose: bool
        whether the true graph needs transposed
    score_type: str
        score functions, 'BIC' or 'BIC_different_var'
    reg_type: str
        regressor type (in combination wth score_type)
    lambda_iter_num: int
        how often to update lambdas
    lambda_flag_default: bool
        with set lambda parameters; true with default strategy and ignore input bounds
    score_bd_tight: bool
        if bound is tight, then simply use a fixed value, rather than the adaptive one
    lambda2_update: float
        increasing  multiplying lambda2
    score_lower: float
        lower bound on lambda1
    score_upper: float
        upper bound on lambda1
    seed: int
        seed
    iterations: int
        nb epoch
    lr1_start: float
        actor learning rate
    lr1_decay_step: int
        lr1 decay step
    lr1_decay_rate: float
        lr1 decay rate
    alpha: float
        update factor moving average baseline
    init_baseline: float
        initial baseline - REINFORCE
    l1_graph_reg: float
        L1 graph regularization to encourage sparsity
    verbose: bool
        print detailed logging or not

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1906.04477

    Examples
    --------
    >>> import castle
    >>> castle.set_backend('mindspore')
    >>> from castle.algorithms import RL
    >>> from castle.datasets import load_dataset
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = RL()
    # If you want to use GPU, you can call CORL().set_context() like the following code:
    >>> algo.set_context(device_target='GPU', mode=ms.GRAPH_MODE) # 'GPU', 'CPU' or 'Ascend'
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    @check_args_value(RL_VALID_PARAMS)
    def __init__(self,
                 encoder_type='TransformerEncoder',
                 hidden_dim=64, 
                 num_heads=16,
                 num_stacks=6, 
                 residual=False, 
                 decoder_type='SingleLayerDecoder', 
                 decoder_activation='tanh', 
                 decoder_hidden_dim=16, 
                 use_bias=False, 
                 use_bias_constant=False, 
                 bias_initial_value=False, 
                 batch_size=64, 
                 input_dimension=64, 
                 normalize=False, 
                 transpose=False, 
                 score_type='BIC', 
                 reg_type='LR', 
                 lambda_iter_num=1000, 
                 lambda_flag_default=True, 
                 score_bd_tight=False,
                 lambda2_update=10, 
                 score_lower=0.0, 
                 score_upper=0.0,
                 seed=8, 
                 iterations=1000,
                 lr1_start=0.001,
                 lr1_decay_step=5000, 
                 lr1_decay_rate=0.96, 
                 alpha=0.99, 
                 init_baseline=-1.0,
                 l1_graph_reg=0.0,
                 verbose=False,
                 ):
        super(RL, self).__init__()

        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.residual = residual
        self.decoder_type = decoder_type
        self.decoder_activation = decoder_activation
        self.decoder_hidden_dim = decoder_hidden_dim
        self.use_bias = use_bias
        self.use_bias_constant = use_bias_constant
        self.bias_initial_value = bias_initial_value
        self.batch_size = batch_size
        self.input_dimension = input_dimension
        self.normalize = normalize
        self.transpose = transpose
        self.score_type = score_type
        self.reg_type = reg_type
        self.lambda_iter_num = lambda_iter_num
        self.lambda_flag_default = lambda_flag_default
        self.score_bd_tight = score_bd_tight
        self.lambda2_update = lambda2_update
        self.score_lower = score_lower
        self.score_upper = score_upper
        self.seed = seed
        self.iterations = iterations
        self.lr1_start = lr1_start
        self.lr1_decay_step = lr1_decay_step
        self.lr1_decay_rate = lr1_decay_rate
        self.alpha = alpha
        self.init_baseline = init_baseline
        self.l1_graph_reg = l1_graph_reg
        self.verbose = verbose

    def learn(self, data, columns=None, dag=None, **kwargs):
        """
        Set up and run the RL algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        dag : ndarray
            two-dimensional, prior matrix
        """
        if ms.get_context('mode') == 0:
            raise ValueError('RL only support pynative mode currently,'
                             ' please set param mode to 1 by using the function set_context!')

        self._set_seed(self.seed)

        self.dag = dag
        X = Tensor(data, columns=columns)
        
        self.data_size = X.shape[0]
        self.max_length = X.shape[1]

        causal_matrix = self._rl(X.data)
        self.causal_matrix = Tensor(causal_matrix, index=X.columns, columns=X.columns)

    def _rl(self, X):
        # Reproducibility

        logging.info('Python version is {}'.format(platform.python_version()))

        # input data
        if self.dag:
            training_set = DataGenerator_read_data(
                X, self.dag, self.normalize, self.transpose)
        else:
            training_set = DataGenerator_read_data(
                X, None, self.normalize, self.transpose)

        # set penalty weights
        score_type = self.score_type
        reg_type = self.reg_type

        if self.lambda_flag_default:
            sl, su, strue = BIC_lambdas(training_set.input_data, None, None, None, reg_type, score_type)
            lambda1 = 0
            lambda1_upper = 5
            lambda1_update_add = 1
            lambda2 = 1/(10**(np.round(self.max_length/3)))
            lambda2_upper = 0.01
            lambda2_update_mul = 10
            lambda_iter_num = self.lambda_iter_num

            # test initialized score
            logging.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
            logging.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(
                sl, su, lambda2, (strue-sl)/(su-sl)*lambda1_upper))
        else:
            # test choices for the case with mannualy provided bounds
            # not fully tested
            sl = self.score_lower
            su = self.score_upper
            if self.score_bd_tight:
                lambda1 = 2
                lambda1_upper = 2
                lambda1_update_add = 2
            else:
                lambda1 = 0
                lambda1_upper = 5
                lambda1_update_add = 1
            lambda2 = 1/(10**(np.round(self.max_length/3)))
            lambda2_upper = 0.01
            lambda2_update_mul = self.lambda2_update
            lambda_iter_num = self.lambda_iter_num

        # actor
        actor = Actor(encoder_type=self.encoder_type,
                      hidden_dim=self.hidden_dim,
                      max_length=self.max_length,
                      num_heads=self.num_heads,
                      num_stacks=self.num_stacks,
                      residual=self.residual,
                      decoder_type=self.decoder_type,
                      decoder_activation=self.decoder_activation,
                      decoder_hidden_dim=self.decoder_hidden_dim,
                      use_bias=self.use_bias,
                      use_bias_constant=self.use_bias_constant,
                      bias_initial_value=self.bias_initial_value,
                      batch_size=self.batch_size,
                      input_dimension=self.input_dimension,
                      lr1_start=self.lr1_start,
                      lr1_decay_step=self.lr1_decay_step,
                      lr1_decay_rate=self.lr1_decay_rate,
                      alpha=self.alpha,
                      init_baseline=self.init_baseline)
        callreward = GetReward(self.batch_size, self.max_length,
                               self.input_dimension, training_set.input_data,
                               sl, su, lambda1_upper, score_type, reg_type,
                               self.l1_graph_reg, False)
        logging.info('Finished creating training dataset and reward class')
        logging.info('Starting training.')

        # Package training information
        net_loss = NetWithLossCell(actor, callreward, lambda1, lambda2)
        train_cell = nn.TrainOneStepCell(net_loss, actor.opt1)
        train_cell.set_train(mode=True)
        
        for i in tqdm(range(1, self.iterations + 1)):

            if self.verbose:
                logging.info('Start training for {}-th epoch'.format(i))

            input_batch = training_set.train_batch(self.batch_size, self.input_dimension)
            inputs = ms.numpy.array(input_batch, dtype=ms.float32)

            # Test tensor shape
            if i == 1:
                logging.info('Shape of actor.input: {}'.format(inputs.shape))

            # actor
            train_cell(inputs)

            if self.verbose:
                logging.info('Finish calculating reward for current batch of graph')

            if self.verbose:
                logging.info('Finish updating actor and critic network using reward calculated')

            # update lambda1, lamda2
            if i == 1 or i % lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores(lambda1, lambda2)
                graph_int, score_min, cyc_min = np.int64(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                if cyc_min < 1e-5:
                    lambda1_upper = score_min
                lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                net_loss.lambda1 = lambda1
                net_loss.lambda2 = lambda2
                logging.info('[iter {}] lambda1 {:.4}, upper {:.4}, lambda2 {:.4}, upper {:.4}, score_min {:.4}, cyc_min {:.4}'.format(
                    i, lambda1*1.0, lambda1_upper*1.0, lambda2*1.0, lambda2_upper*1.0, score_min*1.0, cyc_min*1.0))

                graph_batch = convert_graph_int_to_adj_mat(graph_int)

                if reg_type == 'LR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.input_data))
                elif reg_type == 'QR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.input_data))
                else:
                    raise ValueError("Parameter 'reg_type' only support 'LR' and 'QR'.")

                if self.dag:
                    met = StructureMetrics(graph_batch.T, training_set.true_graph)
                    met2 = StructureMetrics(graph_batch_pruned.T, training_set.true_graph)
                    acc_est = met.metrics
                    acc_est2 = met2.metrics

                    fdr, tpr, fpr, shd, nnz = \
                        acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], \
                        acc_est['shd'], acc_est['nnz']
                    fdr2, tpr2, fpr2, shd2, nnz2 = \
                        acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], \
                        acc_est2['shd'], acc_est2['nnz']

                    logging.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                    logging.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))
        logging.info('Training COMPLETED !')

        return graph_batch_pruned.T
