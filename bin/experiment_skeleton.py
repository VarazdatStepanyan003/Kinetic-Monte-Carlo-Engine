from numba import njit
# from engine import ...


@njit(nogil=True)
def run():
    return None  # result of experiment


def post(res):
    return None  # visualizations, calculations etc (doesn't need to return anything)
