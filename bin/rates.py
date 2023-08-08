from numba import njit, f8
from numba import types
import numpy as np


@njit(types.containers.Tuple((f8, f8))(f8), nogil=True)
def sigmoid(x):
    return 1 / (1 + np.exp(x)), 1


@njit(types.containers.Tuple((f8, f8))(f8), nogil=True)
def always_down(x):
    if x < 0:
        return 1, 1
    p = np.exp(-x)
    return p, 1
