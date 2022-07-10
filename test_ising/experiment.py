from engine import many_simulate, smooth
from matplotlib.pyplot import plot, show, ylim
from bin.helpers import interpolate
from numba import njit


@njit(nogil=True)
def run():
    n_of_steps = 250000
    max_time = 250
    n_of_repetitions = 128
    dt = 1
    times, observabless = many_simulate(n_of_steps=n_of_steps, max_time=max_time, n_of_repetitions=n_of_repetitions)

    time, m = smooth(max_time, dt, times, observabless[:, :, 0])
    time, e = smooth(max_time, dt, times, observabless[:, :, 1])

    m = interpolate(m)
    e = interpolate(e)
    return time, m, e


def post(res):
    time, m, e = res
    plot(time, m)
    ylim(-1.1, 1.1)
    show()
    plot(time, e)
    ylim(-1.1, 1.1)
    show()
