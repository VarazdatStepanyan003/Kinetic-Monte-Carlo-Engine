from numba import njit
import numpy as np
from bin.helpers import binary_search
from config import INPUTS

n_of_observables = 3


@njit("i4[:]()",nogil=True)
def state_init():
    state_init = np.zeros(100, dtype=np.int32)
    for i in range(INPUTS.SIZE):
        if np.random.random() < INPUTS.INFPROB:
            state_init[i] = 1
    return state_init


@njit(nogil=True)
def observables(state, time):
    S, I, R = 0, 0, 0
    for i in range(INPUTS.SIZE):
        if state[i] == 0:
            S += 1
        elif state[i] == 1:
            I += 1
        else:
            R += 1
    return S / INPUTS.SIZE, I / INPUTS.SIZE, R / INPUTS.SIZE


@njit(nogil=True)
def decide(state, time):
    return rejection_kmc(state, time)
    # return rejection_free_kmc(state, time)


@njit(nogil=True)
def rejection_free_kmc(state, time):
    S, I, R = observables(state, time)
    if R == 1 or I == 0:
        return False, state, 0
    rates = np.zeros(INPUTS.SIZE + 1)
    for i in range(1, INPUTS.SIZE + 1):
        if state[i - 1] == 0:
            dr = I * S * alpha(time)
        elif state[i - 1] == 1:
            dr = I * beta(time)
        else:
            dr = 0
        rates[i] = rates[i - 1] + dr
    if rates[INPUTS.SIZE] == 0:
        return False, state, 0
    x = np.random.uniform(0, rates[INPUTS.SIZE])
    j = binary_search(x, rates)
    nstate = np.copy(state)
    nstate[j] += 1
    u = 1 - np.random.random()
    dt = np.log(1 / u) / rates[INPUTS.SIZE]
    return True, nstate, dt


@njit(nogil=True)
def rejection_kmc(state, time):
    S, I, R = observables(state, time)
    if R == 1 or I == 0:
        return False, state, 0
    i = np.random.randint(0, INPUTS.SIZE)
    dr = 0
    if state[i] == 0:
        dr = I * S * alpha(time)
    elif state[i] == 1:
        dr = I * beta(time)
    r0 = max(I * S * alpha(time), I * beta(time))
    u = 1 - np.random.random()
    dt = np.log(1 / u) / r0 / INPUTS.SIZE
    if dr == 0 or np.random.random() > dr / r0:
        return False, state, dt
    nstate = np.copy(state)
    nstate[i] += 1
    return True, nstate, dt


@njit(nogil=True)
def alpha(time):
    return INPUTS.INFRATE


@njit(nogil=True)
def beta(time):
    return INPUTS.RECRATE
