from bin.helpers import var_input

default = [1000, 5, 1, 0.1, 128, 25000, 120, 1]

c = input("Do you wish to input bulk([Y]/N): ")
print()
if c == '' or c == "Y" or c == "y":
    defstr = ",".join(str(x) for x in default)
    print("Default Values are " + defstr)
    print()
    c = input("Please input 8 variables: ")
    if c == '':
        c = defstr
    c = c.split(",")
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
elif c != 'N' and c != 'n':
    print('Illegal Value')
    exit(-1)
else:
    SIZE = var_input("Population Amount", default[0], int)
    INFPROB = var_input("Initial Percentage of Infected", default[1], float) / 100
    if INFPROB < 0 or INFPROB > 1:
        print("Illegal Value")
        exit(-1)
    INFRATE = var_input("Rate of Infection", default[2], float)
    RECRATE = var_input("Rate of Recovery", default[3], float)

    NREPETITIONS = var_input("Number of Simulations Averaged Over", default[4], int)
    NSTEPS = var_input("Number of Steps", default[5], int)
    MAXTIME = var_input("Maximum Simulation Time", default[6], float)
    DT = var_input("Final Count Time Step", default[7], float)
