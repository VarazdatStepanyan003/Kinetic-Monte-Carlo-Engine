from config import sys_name
from time import time as current_time
from importlib import import_module

experiment = import_module(sys_name + ".experiment")

start_time = current_time()
res = experiment.run()
print("Runtime (s): ", current_time() - start_time)
experiment.post(res)
