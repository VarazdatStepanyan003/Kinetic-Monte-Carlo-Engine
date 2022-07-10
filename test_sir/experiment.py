from engine import many_simulate, smooth
from matplotlib.pyplot import plot, show, ylim
from bin.helpers import interpolate
from numba import njit


@njit(nogil=True)
def run():
    n_of_steps = 25000
    max_time = 300
    n_of_repetitions = 128
    dt = 1
    times, observabless = many_simulate(n_of_steps=n_of_steps, max_time=max_time, n_of_repetitions=n_of_repetitions)

    time, S = smooth(max_time, dt, times, observabless[:, :, 0])
    time, I = smooth(max_time, dt, times, observabless[:, :, 1])
    time, R = smooth(max_time, dt, times, observabless[:, :, 2])

    return time, S, I, R


def post(res):
    time, S, I, R = res
    S = interpolate(S)
    I = interpolate(I)
    R = interpolate(R)

    plot(time, S, "b", time, I, "r", time, R, "g")
    ylim(0, 1.1)
    show()
