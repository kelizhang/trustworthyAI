# coding = utf-8

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ms_function


def matrix_exp(matrix):
    """
    compute matrix exponential

    Parameters
    ----------
    matrix: ms.Tensor

    Returns
    -------
    expm: matrix exponential value of A
    """
    expm_val = msnp.zeros(matrix.shape, dtype=ms.float32)
    eye_mat = msnp.eye(matrix.shape[0], dtype=ms.float32)
    k = ms.Tensor(1.0, ms.float32)

    while msnp.norm(eye_mat, 1) > 0:
        expm_val = expm_val + eye_mat
        eye_mat = msnp.matmul(matrix, eye_mat) / k
        k += 1.0
    return expm_val


def compute_h(matrix):
    """
    Return H(A) = trace(e^{A*A}) - d

    Parameters
    ----------
    matrix: ms.Tensor

    Returns
    -------
    h: ms.Tensor
    """
    d = matrix.shape[0]
    h = ms.numpy.trace(matrix_exp(matrix * matrix)) - d
    return h
