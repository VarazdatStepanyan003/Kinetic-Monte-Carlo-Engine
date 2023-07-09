from numba import njit

rate_constant = None  # number
n_of_observables = None  # number of observables


@njit("i4[:]()", nogil=True)
def state_init():
    return None  # the initial state (array of 4byte ints)


@njit("TUPLE((f8, f8, f8,...))(i4[:], f8)", nogil=True)
def observables(state, time):
    return None  # the observables from state


@njit("f8(i4[:], f8)", nogil=True)
def energy(state, time):
    return None  # the energy of state


@njit("Tuple((b1, i4[:], f8))(i4[:], f8)", nogil=True)
def decide(state, time):
    return None  # boolean, new state, dt
