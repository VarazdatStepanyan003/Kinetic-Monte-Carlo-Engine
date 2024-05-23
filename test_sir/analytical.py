import numpy as np
from matplotlib.pyplot import plot, ylim, show
from scipy.integrate import solve_ivp
from config import INPUTS


def run():
    # Parameters
    alpha = INPUTS.INFRATE  # infection rate
    beta = INPUTS.RECRATE  # recovery rate

    # Initial conditions
    S0 = 1 - INPUTS.INFPROB
    I0 = INPUTS.INFPROB
    R0 = 0.0
    initial_conditions = np.array([S0, I0, R0])

    # Time points
    t_start = 0
    t_end = INPUTS.MAXTIME
    t_points = np.arange(t_start, t_end, INPUTS.DT)

    # SIR model differential equations
    def sir_model(t, y):
        S, I, R = y
        dSdt = -alpha * S * I
        dIdt = alpha * S * I - beta * I
        dRdt = beta * I
        return np.array([dSdt, dIdt, dRdt])

    # Solve the differential equations
    solution = solve_ivp(sir_model, [t_start, t_end], initial_conditions, t_eval=t_points)

    # Extract solutions
    S = solution.y[0]
    I = solution.y[1]
    R = solution.y[2]

    return t_points, S, I, R


def post(res):
    time, S, I, R = res

    plot(time, S, "b", time, I, "r", time, R, "g")
    ylim(0, 1.1)
    show()
