from importlib import import_module
import os

print("Welcome to KMCEngine")

ls = os.listdir()
sys_names = [str for str in ls if str.startswith("test") or str.startswith("project")]

print()
for i in range(len(sys_names)):
    print(str(i) + ":" + sys_names[i])

k = int(input("Please choose a simulation: "))
if k < 0 or k >= len(sys_names):
    print("Illegal Input")
    exit(-1)

print()
sys_name = sys_names[k]

if "inputs.py" in os.listdir(os.path.join(sys_name)):
    print("Please input the variables for the simulation")
    print("for default values input blank")
    print()
    INPUTS = import_module(sys_name + ".inputs")

if "analytical.py" in os.listdir(os.path.join(sys_name)):
    c = input("Do you wish to simulate([Y]/N): ")
    if c == '' or c == "Y" or c == "y":
        state_init = import_module(sys_name + ".system").state_init
        decide = import_module(sys_name + ".system").decide
        calc_observables = import_module(sys_name + ".system").observables
        n_of_observables = import_module(sys_name + ".system").n_of_observables

        run = import_module(sys_name + ".experiment").run
        post = import_module(sys_name + ".experiment").post


        print()
        print("Simulation Initialization Complete")
        print()
    elif c != 'N' and c != 'n':
        print('Illegal Value')
        exit(-1)
    else:
        run = import_module(sys_name + ".analytical").run
        post = import_module(sys_name + ".analytical").post

