from numba import njit

rate_constant = None  # number
n_of_observables = None  # number of observables


@njit(nogil=True)
def state_init():
    return None  # the initial state


@njit(nogil=True)
def observables(state, time):
    return None  # the observables from state


@njit(nogil=True)
def energy(state, time):
    return None  # the energy of state


@njit(nogil=True)
def decide(state, time):
    return None  # boolean, new state, dt
