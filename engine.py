from numba import njit
import numpy as np
import system_skeleton as system


@njit(nogil=True)
def simulate(n_of_steps, max_time):
    time = np.zeros(n_of_steps + 1)
    state = np.copy(system.state_init)
    observables = np.zeros(n_of_steps + 1, system.n_of_observables)
    observables[0] = system.observables(state)
    for i in range(1, n_of_steps + 1):
        if time[i - 1] >= max_time:
            observables[i] = observables[i - 1]
            time[i] = time[i - 1]
            continue
        d, n_state = system.decide(state, time[i])
        if d:
            state = n_state
        observables[i] = system.observables(state)
    return observables


@njit(nogil=True)
def simulate_keepstate(n_of_steps, max_time):
    time = np.zeros(n_of_steps + 1)
    state = np.copy(system.state_init)
    system_size = len(state)
    states = np.zeros(n_of_steps + 1, system_size)
    states[0] = np.copy(state)
    observables = np.zeros(n_of_steps + 1, system.n_of_observables)
    observables[0] = system.observables(state)
    for i in range(1, n_of_steps + 1):
        if time[i - 1] >= max_time:
            observables[i] = observables[i - 1]
            time[i] = time[i - 1]
            continue
        d, n_state = decide(state, time[i])
        if d:
            state = n_state
        states[i] = np.copy(state)
        observables[i] = system.observables(state)
    return states, observables
