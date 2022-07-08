from numba import njit
import numpy as np

state_init = None  # array of numbers
rate_constant = None  # number
n_of_observables = None  # number of observables

@njit(nogil=True)
def observables(state):
    return None  # the observables from state

@njit(nogil=True)
def energy(state):
    return None # the energy of state

@njit(nogil=True)
def decide():
    return None  # boolean, new state
