from numba import njit
# from engine import ...


@njit("Tuple((RESULT OF EXPERIMENT TYPE))()", nogil=True)
def run():
    return None  # result of experiment


def post(res):
    return None  # visualizations, calculations etc (doesn't need to return anything)
