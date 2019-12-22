import numpy as np

N_DAYS = 100
N_FAMILIES = 5000
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

cols = [f"choice_{i}" for i in range(10)]
days = list(range(N_DAYS, 0, -1))
