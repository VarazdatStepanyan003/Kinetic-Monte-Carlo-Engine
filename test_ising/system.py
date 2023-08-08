from numba import njit, f8, i4, b1
from numba import types
import numpy as np
from bin.rates import sigmoid
from config import INPUTS

rate_constant = 0.1
n_of_observables = 2


@njit(i4[:](), nogil=True)
def state_init():
    return np.ones(INPUTS.SIZE, dtype=np.int32)


@njit(types.containers.Tuple((f8, f8))(i4[:], f8), nogil=True)
def observables(state, time):
    m = np.mean(state)
    e = 0
    for i in range(INPUTS.SIZE):
        e += state[i] * state[(i + 1) % INPUTS.SIZE]
    return m, e / INPUTS.SIZE


@njit(f8(f8), nogil=True)
def J(time):
    return 1


@njit(f8(f8), nogil=True)
def h(time):
    return 0.1


@njit(f8(f8), nogil=True)
def beta(time):
    return INPUTS.BETA


@njit(f8(i4[:], f8), nogil=True)
def energy(state, time):
    s = 0
    for i in range(INPUTS.SIZE):
        s += -J(time) * state[i] * state[(i + 1) % INPUTS.SIZE] - h(time) * state[i]
    return s


@njit(i4[:](i4[:], i4), nogil=True)
def swap(state, index):
    nstate = np.copy(state)
    nstate[index] *= -1
    return nstate


@njit(types.containers.Tuple((b1, i4[:], f8))(i4[:], f8), nogil=True)
def decide(state, time):
    nstate = swap(state, np.random.randint(INPUTS.SIZE))
    r, R = sigmoid(beta(time) * (energy(nstate, time) - energy(state, time)))
    u = 1 - np.random.random()
    dt = (np.log(1 / u) * rate_constant / R) / INPUTS.SIZE
    if np.random.uniform(0, R) <= r:
        return True, nstate, dt
    return False, state, dt
