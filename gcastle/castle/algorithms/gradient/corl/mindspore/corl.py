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
import platform
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from .frame import Actor, EpisodicCritic, Reward, DenseCritic
from .frame import score_function as score_func
from .utils.data_loader import DataGenerator
from .utils.graph_analysis import get_graph_from_order, pruning_by_coef
from .utils.graph_analysis import pruning_by_coef_2nd
from castle.common import GradientBasedLearner, Tensor, consts
from castle.common.validator import check_args_value
from castle.common.class_factory import ClassFactory


class NetWithLossCell(nn.Cell):
    """
    Packaging network and loss

    Parameters
    ----------
    actor: object
        Actor network instantiate object
    reward: object
        reward network instantiate object
    critic: object
        critic network instantiate object
    dim_para: tuple
        tuple contains several params that are needed in NetWithLossCell
        include (embed_dim, batch_size, seq_length, reward_mode, alpha, avg_baseline)
    """
    def __init__(self, actor, reward, critic, dim_para):
        super(NetWithLossCell, self).__init__()
        self.actor = actor
        self.reward = reward
        self.critic = critic
        self.max_reward = float('-inf')
        self.max_reward_batch = float(0.)
        self.embed_dim = dim_para[0]
        self.batch_size = dim_para[1]
        self.seq_length = dim_para[2]
        self.reward_mode = dim_para[3]
        self.alpha = dim_para[4]
        self.avg_baseline = dim_para[5]
        self.actor_loss = float(0.)
        self.critic_loss = float(0.)
        self.loss = float(0.)

    def construct(self, input_batch):
        """
        Forward propagation process and calculating loss
        """
        encoder_output = self.actor.encode(input=input_batch)
        decoder_output = self.actor.decode(input=encoder_output)
        actions, mask_scores, s_list, h_list, c_list = decoder_output

        batch_graphs = []
        action_mask_s = []
        for m in range(actions.shape[0]):
            zero_matrix = get_graph_from_order(actions[m].asnumpy())
            action_mask = np.zeros(zero_matrix.shape[0])
            for act in actions[m]:
                action_mask_s.append(action_mask.copy())
                action_mask += np.eye(zero_matrix.shape[0])[act]
            batch_graphs.append(zero_matrix)
        batch_graphs = np.stack(batch_graphs)  # 64*10*10
        action_mask_s = np.stack(action_mask_s)

        # Reward
        reward_output = self.reward.cal_rewards(batch_graphs, actions)
        reward_list, normal_batch_reward, max_reward_batch, td_target = reward_output

        self.max_reward_batch = max_reward_batch

        if self.max_reward < self.max_reward_batch:
            self.max_reward = self.max_reward_batch

        # Critic
        prev_input = s_list.reshape((-1, self.embed_dim))
        prev_state_0 = h_list.reshape((-1, self.embed_dim))
        prev_state_1 = c_list.reshape((-1, self.embed_dim))

        action_mask_ = action_mask_s.reshape((-1, self.seq_length))
        log_softmax = self.actor.decoder.log_softmax(input=prev_input,
                                                     position=actions,
                                                     mask=action_mask_,
                                                     state_0=prev_state_0,
                                                     state_1=prev_state_1)
        log_softmax = log_softmax.reshape((self.batch_size,
                                           self.seq_length)).T
        if self.reward_mode == 'episodic':
            self.critic.predict_env(stats_x=s_list[:, :-1, :])
            self.critic.predict_tgt(stats_y=s_list[:, 1:, :])
            self.critic.soft_replacement()
            td_target = td_target[::-1][:-1]

            self.actor_loss = score_func.episodic_actor_loss(
                td_target=ms.Tensor(td_target, ms.float32),
                prediction_env=self.critic.prediction_env,
                log_softmax=log_softmax
            )
            self.critic_loss = score_func.episodic_critic_loss(
                td_target=ms.Tensor(td_target, ms.float32),
                prediction_env=self.critic.prediction_env
            )
        elif self.reward_mode == 'dense':
            log_softmax = ops.ReduceSum(keep_dims=False)(log_softmax, 0)
            reward_mean = np.mean(normal_batch_reward)
            self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean
            predict_reward = self.critic.predict_reward(encoder_output=encoder_output)

            self.actor_loss = score_func.dense_actor_loss(normal_batch_reward,
                                                          self.avg_baseline,
                                                          predict_reward,
                                                          log_softmax)
            self.critic_loss = score_func.dense_critic_loss(normal_batch_reward,
                                                            self.avg_baseline,
                                                            predict_reward)
        else:
            raise ValueError(f"reward_mode must be one of ['episodic', "
                             f"'dense'], but got {self.reward_mode}.")

        self.loss = (self.actor_loss + self.critic_loss)/2
        return self.loss


@ClassFactory.algorithms.register()
class CORL(GradientBasedLearner):
    """
    Causal discovery with Ordering-based Reinforcement Learning

    A RL- and order-based algorithm that improves the efficiency and scalability
    of previous RL-based approach, contains CORL1 with ``episodic`` reward type
    and CORL2 with ``dense`` reward type``.

    References
    ----------
    https://arxiv.org/abs/2105.06631

    Parameters
    ----------
    batch_size: int, default: 64
        training batch size
    input_dim: int, default: 100
        dimension of input data
    embed_dim: int, default: 256
        dimension of embedding layer output
    normalize: bool, default: False
        whether normalization for input data
    encoder_name: str, default: 'transformer'
        Encoder name, must be one of ['transformer', 'lstm', 'mlp']
    encoder_heads: int, default: 8
        number of multi-head of `transformer` Encoder.
    encoder_blocks: int, default: 3
        blocks number of Encoder
    decoder_name: str, default: 'lstm'
        Decoder name, must be one of ['lstm', 'mlp']
    reward_mode: str, default: 'episodic'
        reward mode, 'episodic' or 'dense',
        'episodic' denotes ``episodic-reward``, 'dense' denotes ``dense-reward``.
    reward_score_type: str, default: 'BIC'
        type of score function, must be one of ['BIC', 'BIC_different_var']
    reward_regression_type: str, default: 'LR'
        type of regression function, must be one of ['LR', 'QR']
    reward_gpr_alpha: float, default: 1.0
        alpha of GPR
    iterations: int, default: 5000
        training times
    lambda_iter_num: int, default: 500
        iterations to print min score
    actor_lr: float, default: 1e-4
        learning rate of Actor network, includes ``encoder`` and ``decoder``.
    critic_lr: float, default: 1e-3
        learning rate of Critic network
    alpha: float, default: 0.99
        alpha for score function, includes ``dense_actor_loss`` and
        ``dense_critic_loss``.
    init_baseline: float, default: -1.0
        initilization baseline for score function, includes ``dense_actor_loss``
        and ``dense_critic_loss``.
    random_seed: int, default: 0
        random seed for all random process

    Examples
    --------
    >>> import os
    >>> os.environ['CASTLE_BACKEND'] = 'mindspore'
    >>> from castle import plot_causal_graph
    >>> from castle.metrics import StructureMetrics
    >>> from castle.datasets import DAG, IIDSimulation, load_dataset
    >>> from castle.algorithms import CORL
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> algo = CORL()
    >>> algo.set_context(device_target='CPU', mode=1)
    >>> algo.learn(X)
    >>> plot_causal_graph(algo.causal_matrix, true_dag)
    >>> met = StructureMetrics(algo.causal_matrix, true_dag)
    >>> print(met.values)
    """

    @check_args_value(consts.CORL_VALID_PARAMS)
    def __init__(self, batch_size=64, input_dim=100, embed_dim=256,
                 normalize=False,
                 encoder_name='transformer',
                 encoder_heads=8,
                 encoder_blocks=3,
                 decoder_name='lstm',
                 reward_mode='episodic',
                 reward_score_type='BIC',
                 reward_regression_type='LR',
                 reward_gpr_alpha=1.0,
                 iterations=5000,
                 lambda_iter_num=500,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 alpha=0.99,  # for score function
                 init_baseline=-1.0,
                 random_seed=0):
        super(CORL, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.normalize = normalize
        self.encoder_name = encoder_name
        self.encoder_heads = encoder_heads
        self.encoder_blocks = encoder_blocks
        self.decoder_name = decoder_name
        self.reward_mode = reward_mode
        self.reward_score_type = reward_score_type
        self.reward_regression_type = reward_regression_type
        self.reward_gpr_alpha = reward_gpr_alpha
        self.iterations = iterations
        self.lambda_iter_num = lambda_iter_num
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha = alpha
        self.random_seed = random_seed

        if reward_mode == 'dense':
            self.avg_baseline = ms.Tensor(init_baseline)
        else:
            self.avg_baseline = ms.Tensor(0.)

    def learn(self, data, columns=None, **kwargs) -> None:
        """
        Set up and run the Causal discovery with Ordering-based Reinforcement
        Learning algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        Other Parameters:
            dag_mask : ndarray
                two-dimensional array with [0, 1], shape = [n_nodes, n_nodes].
                (i, j) indicated element `0` denotes there must be no edge
                between nodes `i` and `j` , the element `1` indicates that
                there may or may not be an edge.
        """
        mode = ms.get_context("mode")
        if mode == 1:
            logging.info('The current mindspore mode is `pynative`, and device_type is {}, '
                         'device_ids is {}'.format(ms.get_context("device_target"), ms.get_context("device_id")))
        else:
            raise NotImplementedError('The mindspore mode is `graph`, corl only support '
                                      'pynative mode currently, please set `mode=1`')

        X = Tensor(data, columns=columns)
        self.seq_length = X.shape[1]  # seq_length == n_nodes
        if X.shape[1] > self.batch_size:
            raise ValueError(f'The `batch_size` must greater than or equal to '
                             f'`n_nodes`, but got '
                             f'batch_size: {self.batch_size}, '
                             f'n_nodes: {self.seq_length}.')
        self.dag_mask = getattr(kwargs, 'dag_mask', None)
        causal_matrix = self._rl_search(X.data)
        self.causal_matrix = Tensor(causal_matrix,
                                    index=X.columns,
                                    columns=X.columns)

    def _rl_search(self, X) -> ms.Tensor:
        """
        Search DAG with ordering-based reinforcement learning

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        """

        self._set_seed(self.random_seed)
        logging.info('Python version is {}'.format(platform.python_version()))

        # generate observed data
        data_generator = DataGenerator(dataset=X,
                                       normalize=self.normalize)
        # Instantiating an Actor
        actor = Actor(input_dim=self.input_dim,
                      embed_dim=self.embed_dim,
                      encoder_blocks=self.encoder_blocks,
                      encoder_heads=self.encoder_heads,
                      encoder_name=self.encoder_name,
                      decoder_name=self.decoder_name)
        # Instantiating an Critic
        if self.reward_mode == 'episodic':
            critic = EpisodicCritic(input_dim=self.embed_dim)
        else:
            critic = DenseCritic(input_dim=self.embed_dim,
                                 output_dim=self.embed_dim)

        # Instantiating an Reward
        reward = Reward(input_data=ops.stop_gradient(data_generator.dataset).asnumpy(),
                        reward_mode=self.reward_mode,
                        score_type=self.reward_score_type,
                        regression_type=self.reward_regression_type,
                        alpha=self.reward_gpr_alpha)
        # Instantiating an Optimizer
        actor.encoder.update_parameters_name("actor_encoder")
        actor.decoder.update_parameters_name("actor_decoder")
        critic.update_parameters_name("critic")
        optimizer = nn.Adam([
            {
                'params': actor.encoder.trainable_params(), 'lr': self.actor_lr
            },
            {
                'params': actor.decoder.trainable_params(), 'lr': self.actor_lr
            },
            {
                'params': critic.trainable_params(), 'lr': self.critic_lr
            }
        ])

        # initial max_reward
        logging.info(f'Shape of input batch: {self.batch_size}, '
                     f'{self.seq_length}, {self.input_dim}')
        logging.info(f'Shape of input batch: {self.batch_size}, '
                     f'{self.seq_length}, {self.embed_dim}')
        logging.info('Starting training.')

        graph_batch_pruned = Tensor(np.ones((self.seq_length,
                                             self.seq_length)) -
                                    np.eye(self.seq_length))
        dim_para = (self.embed_dim, self.batch_size, self.seq_length, self.reward_mode,
                    self.alpha, self.avg_baseline)
        net_loss = NetWithLossCell(actor, reward, critic, dim_para)
        train_cell = nn.TrainOneStepCell(net_loss, optimizer)
        train_cell.set_train(mode=True)

        for i in range(1, self.iterations + 1):
            # generate one batch input
            input_batch = data_generator.draw_batch(batch_size=self.batch_size,
                                                    dimension=self.input_dim)
            train_cell(input_batch)
            if i == 1 or i % consts.LOG_FREQUENCY == 0:
                logging.info('[iter {}] max_reward: {:.4}, '
                             'max_reward_batch: {:.4}'.format(i, net_loss.max_reward,
                                                              net_loss.max_reward_batch))
            if i == 1 or i % self.lambda_iter_num == 0:
                ls_kv = reward.update_all_scores()
                score_min, graph_int_key = ls_kv[0][1][0], ls_kv[0][0]
                logging.info('[iter {}] score_min {:.4}'.format(i, score_min * 1.0))
                graph_batch = get_graph_from_order(graph_int_key,
                                                   dag_mask=self.dag_mask)

                if self.reward_regression_type == 'LR':
                    graph_batch_pruned = pruning_by_coef(
                        graph_batch, ops.stop_gradient(data_generator.dataset).asnumpy()
                    )
                elif self.reward_regression_type == 'QR':
                    graph_batch_pruned = pruning_by_coef_2nd(
                        graph_batch, ops.stop_gradient(data_generator.dataset).asnumpy()
                    )
                else:
                    raise ValueError(f"reward_regression_type must be one of "
                                     f"['LR', 'QR'], but got "
                                     f"{self.reward_regression_type}.")

        return graph_batch_pruned.T
