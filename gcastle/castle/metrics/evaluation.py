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

import copy
import warnings
import numpy as np
import pandas as pd
from ..common.base import Tensor


class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true):
        _msg = f"``MetricsDAG`` is deprecated from current version " \
               f"and will be removed in a future vision, " \
               f"use ``castle.metrics.StructureMetrics`` instead!"
        warnings.warn(_msg, DeprecationWarning, 2)
        if isinstance(B_est, Tensor):
            B_est = B_est.data
        if isinstance(B_true, Tensor):
            B_true = B_true.data
        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = copy.deepcopy(B_est)
        self.B_true = copy.deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true)

    @staticmethod
    def _count_accuracy(B_est, B_true, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        decimal_num: int
            Result decimal numbers.

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        # trans diagonal element into 0
        for i in range(len(B_est)):
            if B_est[i, i] == 1:
                B_est[i, i] = 0
            if B_true[i, i] == 1:
                B_true[i, i] = 0

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == B_est[j, i] == 1:
                    B_est[i, j] = -1
                    B_est[j, i] = 0

        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            # if not is_dag(B_est):
            #     raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]

        # linear index of nonzeros
        pred_und = np.concatenate([np.flatnonzero(B_est == -1), np.flatnonzero(B_est.T == -1)])
        pred = np.flatnonzero(B_est == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.unique(np.concatenate([cond, cond_reversed]))
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        for i in range(len(B_est)):
            for j in range(len(B_est[i])):
                if B_est[i, j] == -1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

        W_p = pd.DataFrame(B_est)
        W_true = pd.DataFrame(B_true)

        gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size,
              'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        for i in mt:
            mt[i] = round(mt[i], decimal_num)

        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        num_true = W_true.sum(axis=1).sum()
        assert num_true != 0

        # true_positives
        if pd.__version__ >= "2.1.0":
            num_tp = (W_p + W_true).map(lambda elem: 1 if elem == 2 else 0).sum(axis=1).sum()
        else:
            num_tp =  (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        # False Positives + Reversed Edges
        if pd.__version__ >= "2.1.0":
            num_fn_r = (W_p - W_true).map(lambda elem: 1 if elem == 1 else 0).sum(axis=1).sum()
        else:
            num_fn_r = (W_p - W_true).applymap(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
        score = np.max((num_tp-num_fn_r,0))/num_true
        
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert(W_p.shape==W_true.shape and W_p.shape[0]==W_p.shape[1])
        if pd.__version__ >= "2.1.0":
            TP = (W_p + W_true).map(lambda elem: 1 if elem == 2 else 0).sum(axis=1).sum()
        else:
            TP = (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP / TP_FP
        recall = TP / TP_FN
        F1 = 2 * (recall * precision) / (recall + precision)

        return precision, recall, F1


class StructureMetrics(object):
    """
    Compute various accuracy metrics for est_graph

    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP / (TP + FP)
    recall: TP / (TP + FN)
    F1: 2 * (recall * precision) / (recall + precision)
    gscore: max(0, (TP - FP)) / (TP + FN), A score ranges from 0 to 1

    Parameters
    ----------
    est_graph: np.ndarray
        [d, d] estimate, {0, 1}.
    true_graph: np.ndarray
        [d, d] ground truth graph, {0, 1}.

    Examples
    --------
    >>> import numpy as np
    >>> from castle.metrics.evaluation import StructureMetrics
    >>> est_graph = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0]])
    >>> true_graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> metrics = StructureMetrics(est_graph, true_graph)
    You can get all metrics from metrics.values, like the following example:
    >>> metrics_value = metrics.values
    >>> print(metrics_value)
    {'fdr': 0.3333, 'tpr': 1.0, 'fpr': 1.0, 'shd': 1, 'nnz': 3, 'precision': 0.6667, 'recall': 1.0, 'F1': 0.8, 'gscore': 0.5}

    Also, you can just get one metrics score if you need:
    >>> f1 = metrics.f1_score
    >>> print(f1)
    0.8
    >>> gscore = metrics.g_score
    >>> print(gscore)
    0.5
    """

    def __init__(self, est_graph, true_graph):
        self._values = None
        self.est_graph = est_graph
        self.true_graph = true_graph

    def _cal_metrics(self):
        """
        Return fdr, tpr, fpr, shd, nnz, precision, recall, F1-score and g-score by pd.DataFrame

        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """
        if self._values is not None:
            return self._values

        est_graph, true_graph = self.check_params(self.est_graph, self.true_graph)
        d = true_graph.shape[0]

        # linear index of nonzeros
        pred_und = np.flatnonzero(est_graph == -1)
        pred = np.flatnonzero(est_graph == 1)
        cond = np.flatnonzero(true_graph)
        cond_reversed = np.flatnonzero(true_graph.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = round((len(reverse) + len(false_pos)) / max(pred_size, 1), 4)
        tpr = round((len(true_pos)) / max(len(cond), 1), 4)
        fpr = round((len(reverse) + len(false_pos)) / max(cond_neg_size, 1), 4)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(est_graph + est_graph.T))
        cond_lower = np.flatnonzero(np.tril(true_graph + true_graph.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        # trans cpdag [-1, 0, 1] to [0, 1], -1 is undirected edge in CPDAG
        est_graph[est_graph == -1] = 1

        gscore = self._cal_gscore(est_graph, true_graph)

        TP = ((est_graph + true_graph) == 2).sum()
        TP_FP = (est_graph == 1).sum()
        TP_FN = (true_graph == 1).sum()

        precision = round(TP / TP_FP, 4)
        recall = round(TP / TP_FN, 4)
        F1 = round(2 * (recall * precision) / (recall + precision), 4)

        self._values = {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size,
                        'precision': precision, 'recall': recall, 'F1': F1, 'gscore': gscore}
        return self._values

    @staticmethod
    def check_params(est_graph, true_graph):
        """
        Check parameters and make them available

        Parameters
        ----------
        est_graph: np.ndarray, Tensor
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        true_graph: np.ndarray, Tensor
            [d, d] ground truth graph, {0, 1}.

        Returns
        -------
        est_graph
        true_graph
        """

        if isinstance(est_graph, Tensor):
            est_graph = est_graph.data
        if isinstance(true_graph, Tensor):
            true_graph = true_graph.data

        est_graph = np.array(est_graph - np.diag(np.diag(est_graph)))
        true_graph = np.array(true_graph - np.diag(np.diag(true_graph)))

        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        for i in range(len(est_graph)):
            for j in range(len(est_graph[i])):
                if est_graph[i, j] == est_graph[j, i] == 1:
                    est_graph[i, j] = -1
                    est_graph[j, i] = 0
        if (est_graph == -1).any():  # cpdag
            if not ((est_graph == 0) | (est_graph == 1) | (est_graph == -1)).all():
                raise ValueError('est_graph should take value in {0, 1, -1}.')
            if ((est_graph == -1) & (est_graph.T == -1)).any():
                raise ValueError('undirected edge should only appear once.')
        else:  # dag
            if not ((est_graph == 0) | (est_graph == 1)).all():
                raise ValueError('est_graph should take value in {0, 1}.')
        return est_graph, true_graph

    @staticmethod
    def _cal_gscore(est_graph, true_graph):
        """
        Return g-score

        Parameters
        ----------
        est_graph: array
            [d, d] estimate graph, {0, 1}.
        true_graph: array
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        assert not set(np.unique(true_graph)) - {0, 1}
        num_true = (true_graph == 1).sum()
        assert num_true != 0
        assert not set(np.unique(est_graph)) - {0, 1}
        # true_positives
        num_tp = ((est_graph + true_graph) == 2).sum()
        # False Positives + Reversed Edges
        num_fn_r = ((est_graph - true_graph) == 1).sum()
        score = np.max((num_tp - num_fn_r, 0)) / num_true

        return round(score, 4)

    @property
    def values(self):
        return self._cal_metrics()

    @property
    def g_score(self):
        return self.values['gscore']

    @property
    def precision(self):
        return self.values['precision']

    @property
    def f1_score(self):
        return self.values['F1']

    @property
    def recall(self):
        return self.values['recall']

    @property
    def shd(self):
        return self.values['shd']

    @property
    def nnz(self):
        return self.values['nnz']

    @property
    def fpr(self):
        return self.values['fpr']

    @property
    def tpr(self):
        return self.values['tpr']

    @property
    def fdr(self):
        return self.values['fdr']
