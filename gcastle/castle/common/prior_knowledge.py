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


class PriorKnowledge(object):
    """
    A class for a prior knowledge.

    Parameters
    ----------
    n_nodes: int
        denotes the number of nodes

    Attributes
    ----------
    matrix: np.ndarray
        0  : i does not have a directed path to j;
        1  : i has a directed path to j;
        -1 : No prior background_knowledge is available to know if either of
             the two cases above (0 or 1) is true.

    Examples
    --------
    >>> prior = PriorKnowledge(4)
    >>> prior.add_required_edges([(0, 1), (1, 3)])
    >>> prior.add_forbidden_edges([(1, 2), (2, 3)])
    >>> prior.add_undirected_edge(0, 3)
    >>> print(prior.matrix)
    An more usually use case is to create an instance from a prior matrix:
    >>> prior_knowledge = PriorKnowledge.from_prior_matrix(prior.matrix)
    >>> print(prior_knowledge.matrix)
    """

    def __init__(self, n_nodes) -> None:
        self.matrix = (np.zeros((n_nodes, n_nodes), dtype=int)
                       - np.ones((n_nodes, n_nodes), dtype=int)
                       + np.eye(n_nodes, dtype=int))
        self.forbidden_edges = []
        self.required_edges = []

    def add_required_edge(self, i, j) -> None:
        """Add a required edge `i-->j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_required_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [-1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 1
        self.required_edges.append((i, j))

    def add_required_edges(self, edges) -> None:
        """Add multiple required edges using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_required_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [-1  0  1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_required_edge(i, j)

    def add_forbidden_edge(self, i, j) -> None:
        """Add a forbidden edge between `i` and `j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_forbidden_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  0 -1 -1]
         [-1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 0
        self.forbidden_edges.append((i, j))

    def add_forbidden_edges(self, edges) -> None:
        """Add multiple forbidden edges using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_forbidden_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  0 -1 -1]
         [-1  0  0 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_forbidden_edge(i, j)

    def add_undirected_edge(self, i, j) -> None:
        """Add an edge with unknown direction `i---j`.

        Parameters
        ----------
        i: int
            denotes location of the source node
        j: int
            denotes location of the target node

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_undirected_edge(0, 1)
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [ 1  0 -1 -1]
         [-1 -1  0 -1]
         [-1 -1 -1  0]]
        """

        self.matrix[i, j] = 1
        self.matrix[j, i] = 1
        self.add_required_edges([(i, j), (j, i)])

    def add_undirected_edges(self, edges) -> None:
        """Add multiple edges with unknown direction by using a list of tuples.

        Parameters
        ----------
        edges: list
            list of (i, j)

        Examples
        --------
        >>> p = PriorKnowledge(4)
        >>> p.add_undirected_edges([(0, 1), (1, 2)]
        >>> print(p.matrix)
        [[ 0  1 -1 -1]
         [ 1  0  1 -1]
         [-1  1  0 -1]
         [-1 -1 -1  0]]
        """

        for i, j in edges:
            self.add_undirected_edge(i, j)

    def is_forbidden(self, i, j) -> bool:
        return (i, j) in self.forbidden_edges

    def is_required(self, i, j) -> bool:
        return (i, j) in self.required_edges

    def remove_edge(self, i, j) -> None:
        """Remove the edge (i, j)

        Make (i, j) convert to unknown edge no matter whether required or forbidden.
        """

        if (i, j) in self.required_edges:
            self.required_edges.remove((i, j))
        if (i, j) not in self.forbidden_edges:
            self.forbidden_edges.append((i, j))
        self.matrix[i, j] = -1

    @staticmethod
    def from_prior_matrix(prior_matrix):
        """
        Return an instance of PriorKnowledge

        Parameters
        ----------
        prior_matrix: array, [n_nodes, n_nodes]
            0  : i does not have a directed path to j;
            1  : i has a directed path to j;
            -1 : No prior background_knowledge is available to know if either of
                 the two cases above (0 or 1) is true.
        """

        output = PriorKnowledge(prior_matrix.shape[0])
        assert not (set(np.unique(prior_matrix)) - {-1, 0, 1}), \
            f"The element value of graph_matrix must be one of [-1, 0, 1]."
        for i, j in zip(*np.where(prior_matrix == 0)):
            if i == j:
                continue
            output.add_forbidden_edge(i, j)
        for i, j in zip(*np.where(prior_matrix == 1)):
            if i == j:
                continue
            output.add_required_edge(i, j)
        return output


def orient_by_prior_knowledge(skeleton, prior_knowledge):
    """
    Orient the direction of all edges based on a prior knowledge.

    Parameters
    ----------
    skeleton: np.ndarray
        a skeleton matrix
    prior_knowledge: PriorKnowledge
        a class object

    Returns
    -------
    out: np.ndarray
        where out[i, j] = out[j, i] = 0 indicates `i   j`;
        out[i, j] = 1 and out[j, i] = 0 indicates `i-->j`;
        out[i, j] = 1 and out[j, i] = 1 indicates `i---j`;
    """

    for i, j in prior_knowledge.required_edges:
        if skeleton[i, j] == 0:
            skeleton[i, j] = 1
    for i, j in prior_knowledge.forbidden_edges:
        if skeleton[i, j] == 1:
            skeleton[i, j] = 0
    return skeleton
