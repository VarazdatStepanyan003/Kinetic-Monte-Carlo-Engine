from numba import njit
import numpy as np


@njit(nogil=True)
def sigmoid(x):
    return 1 / (1 + np.exp(x)), 1 / (1 + np.exp(x)) + 1 / (1 + np.exp(-x))


@njit(nogil=True)
def always_down(x):
    if x < 0:
        return 1, 1
    p = np.exp(-x)
    return p, 1
