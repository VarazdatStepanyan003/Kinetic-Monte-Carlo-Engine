from numba import njit
import numpy as np
from bin.rates import sigmoid
from config import INPUTS

rate_constant = 0.1
n_of_observables = 2


@njit(nogil=True)
def state_init():
    return np.ones(INPUTS.SIZE)


@njit(nogil=True)
def observables(state, time):
    m = np.mean(state)
    e = 0
    for i in range(INPUTS.SIZE):
        e += state[i] * state[(i + 1) % INPUTS.SIZE]
    return m, e / INPUTS.SIZE


@njit(nogil=True)
def energy(state, time):
    s = 0
    for i in range(INPUTS.SIZE):
        s += -J(time) * state[i] * state[(i + 1) % INPUTS.SIZE] - h(time) * state[i]
    return s


@njit(nogil=True)
def decide(state, time):
    nstate = swap(state, np.random.randint(INPUTS.SIZE))
    r, R = sigmoid(beta(time) * (energy(nstate, time) - energy(state, time)))
    u = 1 - np.random.random()
    dt = (np.log(1 / u) * rate_constant / R) / INPUTS.SIZE
    if np.random.uniform(0, R) <= r:
        return True, nstate, dt
    return False, state, dt


@njit(nogil=True)
def swap(state, index):
    nstate = np.copy(state)
    nstate[index] *= -1
    return nstate


@njit(nogil=True)
def J(time):
    return 1


@njit(nogil=True)
def h(time):
    return 0.1


@njit(nogil=True)
def beta(time):
    return INPUTS.BETA
