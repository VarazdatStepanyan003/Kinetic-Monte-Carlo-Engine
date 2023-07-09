from numba import njit
from math import floor
import numpy as np


def var_input(prompt, default, dtype):
    c = input(prompt + "[" + str(default) + "]" ": ")
    if c == '':
        return default
    return dtype(c)


@njit(["i4(i4, i4)", "i4(f8, f8)"], nogil=True)
def kron_delta(a, b):
    if a == b:
        return 1
    return 0


@njit("i4(f8, f8[:])", nogil=True)
def binary_search(x, arr):
    a = 0
    b = len(arr) - 1
    while a < b:
        m = floor((a + b) / 2)
        if arr[m + 1] <= x:
            a = m + 1
        elif arr[m] > x:
            b = m
        else:
            while m > 0 and arr[m - 1] == x:
                m -= 1
            return m
    return -1


@njit("b1[:](f8[:])", nogil=True)
def nan_helper(y):
    return np.isnan(y)


@njit("f8[:](f8[:])", nogil=True)
def interpolate(y):
    nans = nan_helper(y)
    x = lambda z: z.nonzero()[0]
    y_new = np.copy(y)
    y_new[nans] = np.interp(x(nans), x(~nans), y_new[~nans])
    return y_new


@njit("f8[:](f8[:])", nogil=True)
def fill(y):
    y_n = np.copy(y)
    for i in range(len(y)):
        if y_n[i] == np.NaN:
            y_n[i] = y[i - 1]
    return y_n
