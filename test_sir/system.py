from numba import njit
import numpy as np
from bin.helpers import binary_search

state_init = np.zeros(100)
N = len(state_init)
for i in range(N):
    if np.random.random() < 0.05:
        state_init[i] = 1
n_of_observables = 3


@njit(nogil=True)
def observables(state, time):
    S, I, R = 0, 0, 0
    for i in range(N):
        if state[i] == 0:
            S += 1
        elif state[i] == 1:
            I += 1
        else:
            R += 1
    return S / N, I / N, R / N


@njit(nogil=True)
def decide(state, time):
    S, I, R = observables(state, time)
    if R == 1 or I == 0:
        return False, state, 0
    rates = np.zeros(N + 1)
    for i in range(1, N + 1):
        if state[i - 1] == 0:
            dr = I * S * alpha(time)
        elif state[i - 1] == 1:
            dr = I * beta(time)
        else:
            dr = 0
        rates[i] = rates[i - 1] + dr
    if rates[N] == 0:
        return False, state, 0
    x = np.random.uniform(0, rates[N])
    j = binary_search(x, rates)
    nstate = np.copy(state)
    nstate[j] += 1
    u = 1 - np.random.random()
    dt = np.log(1 / u) / rates[N - 1]
    return True, nstate, dt


@njit(nogil=True)
def alpha(time):
    return 1


@njit(nogil=True)
def beta(time):
    return 0.1
