from engine import many_simulate, smooth
from matplotlib.pyplot import plot, show, ylim
from bin.helpers import interpolate
from numba import njit, types, f8
from config import INPUTS


@njit(types.containers.Tuple((f8[:], f8[:], f8[:]))(), nogil=True)
def run():
    times, observabless = many_simulate(n_of_steps=INPUTS.NSTEPS, max_time=INPUTS.MAXTIME,
                                        n_of_repetitions=INPUTS.NREPETITIONS)

    time, m = smooth(INPUTS.MAXTIME, INPUTS.DT, times, observabless[:, :, 0])
    time, e = smooth(INPUTS.MAXTIME, INPUTS.DT, times, observabless[:, :, 1])

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
