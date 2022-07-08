from engine import many_simulate, smooth
from matplotlib.pyplot import plot, show, xlabel, ylabel, ylim
from time import time as current_time
from helpers import interpolate

start_time = current_time()

n_of_steps = 250000
max_time = 5000
n_of_repetitions = 128
dt = 1
times, observabless = many_simulate(n_of_steps=n_of_steps, max_time=max_time, n_of_repetitions=n_of_repetitions)

time, m = smooth(max_time, dt, times, observabless[:, :, 0])
time, e = smooth(max_time, dt, times, observabless[:, :, 1])

m = interpolate(m)
e = interpolate(e)

print(current_time() - start_time)

plot(time, m)
ylim(-1.1, 1.1)
show()
plot(time, e)
ylim(-1.1, 1.1)
show()
