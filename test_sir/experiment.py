from engine import many_simulate, smooth
from matplotlib.pyplot import plot, show, ylim
from bin.helpers import interpolate
from numba import njit
from config import INPUTS


@njit(nogil=True)
def run():
    times, observabless = many_simulate(n_of_steps=INPUTS.NSTEPS, max_time=INPUTS.MAXTIME,
                                        n_of_repetitions=INPUTS.NREPETITIONS)

    time, S = smooth(INPUTS.MAXTIME, INPUTS.DT, times, observabless[:, :, 0])
    time, I = smooth(INPUTS.MAXTIME, INPUTS.DT, times, observabless[:, :, 1])
    time, R = smooth(INPUTS.MAXTIME, INPUTS.DT, times, observabless[:, :, 2])

    return time, S, I, R


def post(res):
    time, S, I, R = res
    S = interpolate(S)
    I = interpolate(I)
    R = interpolate(R)

    plot(time, S, "b", time, I, "r", time, R, "g")
    ylim(0, 1.1)
    show()
