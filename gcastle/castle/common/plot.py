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

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from .base import Tensor


class GraphDAG(object):
    '''
    Visualization for causal discovery learning results.

    .. deprecated ::
        Use castle.plot_causal_graph() instead
    Parameters
    ----------
    est_dag: np.ndarray
        The DAG matrix to be estimated.
    true_dag: np.ndarray
        The true DAG matrix.
    show: bool
        Select whether to display pictures.
    save_name: str
        The file name of the image to be saved.
    '''

    def __init__(self, est_dag, true_dag=None, show=True, save_name=None):
        _msg = f"``GraphDAG`` is deprecated from current version " \
               f"and will be removed in a future version, use ``castle.plot_causal_graph`` instead!"
        warnings.warn(_msg, DeprecationWarning, 2)
        if isinstance(est_dag, Tensor):
            est_dag = est_dag.data
        if isinstance(true_dag, Tensor):
            true_dag = true_dag.data
        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name

        if not isinstance(est_dag, np.ndarray):
            raise TypeError("Input est_dag is not numpy.ndarray!")

        if true_dag is not None and not isinstance(true_dag, np.ndarray):
            raise TypeError("Input true_dag is not numpy.ndarray!")

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        GraphDAG._plot_dag(self.est_dag, self.true_dag, self.show, self.save_name)

    @staticmethod
    def _plot_dag(est_dag, true_dag, show=True, save_name=None):
        """
        Plot the estimated DAG and the true DAG.

        Parameters
        ----------
        est_dag: np.ndarray
            The DAG matrix to be estimated.
        true_dag: np.ndarray
            The True DAG matrix.
        show: bool
            Select whether to display pictures.
        save_name: str
            The file name of the image to be saved.
        """

        if isinstance(true_dag, np.ndarray):

            # trans diagonal element into 0
            for i in range(len(true_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0
                if true_dag[i][i] == 1:
                    true_dag[i][i] = 0

            # set plot size
            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            ax1.set_title('est_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            ax2.set_title('true_graph')
            map2 = ax2.imshow(true_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map2, ax=ax2)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        elif isinstance(est_dag, np.ndarray):

            # trans diagonal element into 0
            for i in range(len(est_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0

            # set plot size
            fig, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

            ax1.set_title('est_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()


def plot(save_path=None, show=True, **key_graph):
    """
    Plot some graph in one figure

    Parameters
    ----------
    file_path: str
        If not None, save the figure to the file path.
    show: bool, default: True
        Whether show the figure
    key_graph: keyword parameters
        The key is the title of a graph, the value is a graph.
        e.g. true_graph=graph, means that the title of the graph is "true_graph" and the value
        is a graph that is numpy.ndarray

    Examples
    --------
    >>> import numpy as np

    >>> graph_1 = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])
    >>> graph_2 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> graph_3 = np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]])

    You can use plot like the following code:
    >>> plot(graph1=graph_1, graph_2=graph_2, graph_3=graph_3)
    where the key of keyword parameters is denotes the title of each subplot.

    Also, if you need to save the figure to your disk, you can do like the following:
    >>> plot(save_path='./figure.png', graph1=graph_1, graph_2=graph_2, graph_3=graph_3)
    """

    num = len(key_graph)
    cols_num = int(np.ceil(np.sqrt(num)))
    rows_num = int(np.ceil(num / cols_num))
    fig, axes = plt.subplots(figsize=(4 * cols_num, 3 * rows_num),
                             ncols=cols_num, nrows=rows_num)
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(-1)
    else:
        axes = np.array([axes])
    for i, k in enumerate(key_graph):
        graph = key_graph.get(k)
        if isinstance(graph, Tensor):
            graph = graph.data
        assert isinstance(graph, np.ndarray), \
            "The type of graph must be numpy.ndarry or castle.Tensor ."
        for j in range(len(graph)):
            graph[j][j] = 0
        axes[i].set_title(k)
        map1 = axes[i].imshow(graph, cmap='Greys', interpolation='none')
        fig.colorbar(map1, ax=axes[i])
    for ax in axes[num:]:
        plt.delaxes(ax)
    plt.subplots_adjust(left=0.05, right=0.92, hspace=0.5)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path)
    if show:
        plt.show()


def plot_causal_graph(est_graph, true_graph=None, save_path=None, show=True):
    """
    Plot estimated graph and true graph

    Parameters
    ----------
    est_graph: Tensor, np.ndarray
        The estimated graph
    true_graph: Tensor, np.ndarray
        The true graph
    save_path: str
        If not None, save the figure to the file path.
    show: bool, default: True
        Whether show the figure

    Examples
    --------
    >>> import numpy as np

    >>> est_graph = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])
    >>> true_graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> plot_causal_graph(est_graph, true_graph, show=True)

    If you need save the figure, you can pass in `file_path` parameter:
    >>> plot_causal_graph(est_graph, true_graph, file_path='./figure.png', show=True)
    """

    if true_graph is not None:
        plot(save_path=save_path, show=show, est_graph=est_graph, true_graph=true_graph)
    else:
        plot(save_path=save_path, show=show, est_graph=est_graph)
