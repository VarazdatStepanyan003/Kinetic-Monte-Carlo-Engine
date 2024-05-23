from config import run, post
from time import time as current_time

start_time = current_time()
res = run()
print("Runtime (s): ", current_time() - start_time)
post(res)
