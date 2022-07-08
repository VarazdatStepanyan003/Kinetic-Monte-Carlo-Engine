from numba import njit


@njit(nogil=True)
def kron_delta(a, b):
    if a == b:
        return 1
    return 0
