from numba import njit, prange
import numpy as np
from config import SYSTEM
from bin.helpers import binary_search


@njit("Tuple((f8[:], f8[:,:]))(i4, f8)", nogil=True)
def simulate(n_of_steps, max_time):
    time = np.zeros(n_of_steps + 1)
    state = SYSTEM.state_init()
    observables = np.zeros((n_of_steps + 1, SYSTEM.n_of_observables))
    observables[0] = SYSTEM.observables(np.copy(state), 0)
    for i in range(1, n_of_steps + 1):
        if (not max_time == -1) and time[i - 1] >= max_time:
            observables[i] = observables[i - 1]
            time[i] = time[i - 1]
            continue
        d, n_state, dt = SYSTEM.decide(np.copy(state), time[i])
        if d:
            state = np.copy(n_state)
            observables[i] = SYSTEM.observables(np.copy(state), time[i])
        else:
            observables[i] = observables[i - 1]
        time[i] = time[i - 1] + dt
    return time, observables


@njit("Tuple((f8[:], f8[:,:], i4[:,:]))(i4, f8)", nogil=True)
def simulate_keepstate(n_of_steps, max_time):
    time = np.zeros(n_of_steps + 1)
    state = SYSTEM.state_init()
    observables = np.zeros((n_of_steps + 1, SYSTEM.n_of_observables))
    observables[0] = SYSTEM.observables(np.copy(state), 0)
    SYSTEM_size = len(state)
    states = np.zeros((n_of_steps + 1, SYSTEM_size), dtype=np.int32)
    states[0] = np.copy(state)
    for i in range(1, n_of_steps + 1):
        if (not max_time == -1) and time[i - 1] >= max_time:
            observables[i] = observables[i - 1]
            time[i] = time[i - 1]
            state[i] = state[i - 1]
            continue
        d, n_state, dt = SYSTEM.decide(np.copy(state), time[i])
        if d:
            state = np.copy(n_state)
            observables[i] = SYSTEM.observables(np.copy(state), time[i])
            states[i] = np.copy(state)
        else:
            observables[i] = observables[i - 1]
            states[i] = states[i - 1]
        time[i] = time[i - 1] + dt
    return time, observables, states


@njit("Tuple((f8[:,:], f8[:,:,:]))(i4, f8, i4)", parallel=True, nogil=True)
def many_simulate(n_of_steps, max_time, n_of_repetitions):
    times = np.zeros((n_of_repetitions, n_of_steps + 1))
    observabless = np.zeros((n_of_repetitions, n_of_steps + 1, SYSTEM.n_of_observables))
    for i in prange(n_of_repetitions):
        times[i], observabless[i] = simulate(n_of_steps, max_time)
    return times, observabless


@njit("Tuple((f8[:], f8[:]))(f8, f8, f8[:,:], f8[:,:])", nogil=True)
def smooth(max_time, dt, times, valuess):
    if max_time == -1:
        max_time = np.max(times[:, len(times[0]) - 1])
    time = np.arange(0, max_time + dt, dt)
    values = np.zeros(len(time))
    k = np.zeros(len(time))
    for i in range(len(valuess)):
        for j in range(len(valuess[0])):
            ind = binary_search(times[i][j], time)
            k[ind] += 1
            values[ind] += valuess[i][j]
    for i in range(len(time)):
        if k[i] != 0:
            values[i] /= k[i]
        else:
            values[i] = np.NaN
    return time, values
