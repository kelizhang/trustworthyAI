# coding = utf-8

from itertools import permutations
import numpy as np
from scipy import stats
from statsmodels.gam.api import GLMGam, BSplines


def estimate_weight(data, method='gaussian'):
    d = data.shape[1]
    w = np.zeros((d, d))
    for i, j in permutations(range(d), 2):
        xi = data[:, i]
        xj = data[:, [j]]
        fi = gam(xi, xj)
        if method == 'gaussian':
            w[j, i] = 0.5 * np.log(np.var(xi - fi) / np.var(xi))
        elif method == 'entropy':
            w[j, i] = stats.differential_entropy(xi - fi) - stats.differential_entropy(xi)
    return w


def gam(x, y):
    sm = BSplines(x, df=[10], degree=[4])
    gam_model = GLMGam(endog=y, smoother=sm)
    gam_model_res = gam_model.fit()
    return gam_model_res.resid_response
