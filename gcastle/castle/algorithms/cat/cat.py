# coding = utf-8

import networkx as nx
from networkx.algorithms.tree import branchings

from castle.common.class_factory import ClassFactory
from castle.common.base import BaseLearner, Tensor
from .functionals.regression import estimate_weight


@ClassFactory.algorithms.register()
class Cat(BaseLearner):
    """
    Causal Additive Trees

    References
    ----------
    https://arxiv.org/pdf/2108.08871.pdf
    """
    def __init__(self, noise='gaussian'):
        super(Cat, self).__init__()
        self.noise = noise

    def learn(self, data, columns=None, **kwargs):
        x = Tensor(data, columns=columns)
        if self.noise == 'gaussian':
            w = estimate_weight(x.data, method='gaussian')
        elif self.noise == 'entropy':
            w = estimate_weight(x.data, method='entropy')
        else:
            raise ValueError(f"The noise must be one of ['gaussian', 'entropy'], "
                             f"but got {self.noise}.")
        g = nx.from_numpy_array(w, create_using=nx.DiGraph)
        mst = branchings.minimum_spanning_arborescence(g)
        self.causal_matrix = Tensor((nx.to_numpy_array(mst) != 0) * 1,
                                    index=x.columns,
                                    columns=x.columns)
