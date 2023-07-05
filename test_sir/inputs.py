from bin.helpers import var_input

c = input("Do you wish to input bulk([Y]/N): ")
print()
if c == '' or c == "Y" or c == "y":
    c = input("Please input 8 variables: ").split(",")
    if len(c) != 8:
        print("Illegal Value")
        exit(-1)
    SIZE = int(c[0])
    INFPROB = float(c[1]) / 100
    if INFPROB < 0 or INFPROB > 1:
        print("Illegal Value")
        exit(-1)
    INFRATE = float(c[2])
    RECRATE = float(c[3])

    NREPETITIONS = int(c[4])
    NSTEPS = int(c[5])
    MAXTIME = float(c[6])
    DT = float(c[7])
elif c != 'N' or c != 'n':
    print('Illegal Value')
    exit(-1)
else:
    SIZE = var_input("Population Amount", 100, int)
    INFPROB = var_input("Initial Percentage of Infected", 5, float) / 100
    if INFPROB < 0 or INFPROB > 1:
        print("Illegal Value")
        exit(-1)
    INFRATE = var_input("Rate of Infection", 1, float)
    RECRATE = var_input("Rate of Recovery", 0.1, float)

    NREPETITIONS = var_input("Number of Simulations Averaged Over", 128, int)
    NSTEPS = var_input("Number of Steps", 25000, int)
    MAXTIME = var_input("Maximum Simulation Time", 300, float)
    DT = var_input("Final Count Time Step", 1, float)
# 100,7,0.1,1,128,50000,-1,0.1