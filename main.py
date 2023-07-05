from config import EXPERIMENT
from time import time as current_time

start_time = current_time()
res = EXPERIMENT.run()
print("Simulation Runtime (s): ", current_time() - start_time)
EXPERIMENT.post(res)
