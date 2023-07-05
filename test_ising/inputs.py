from bin.helpers import var_input

SIZE = var_input("Number of Spins", 100, int)
NREPETITIONS = var_input("Number of Simulations Averaged Over", 128, int)
NSTEPS = var_input("Number of Steps", 250000, int)
MAXTIME = var_input("Maximum Simulation Time", -1, float)
DT = var_input("Final Count Time Step", 1, float)
BETA = var_input("Inverse Temperature", 1, float)
