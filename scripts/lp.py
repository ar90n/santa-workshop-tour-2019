import numpy as np
import pandas as pd
import random
from ortools.linear_solver import pywraplp
from santa_workshop_tour_2019.lp_mip import solveSantaIP, solveSantaLP

np.random.seed(2019)

N_DAYS = 5
N_FAMILIES = 10
FAMILY_SIZE = np.random.choice(4, N_FAMILIES) + 1

NUM_FAMILY_MEMBER = FAMILY_SIZE.sum()
MIN_OCCUPANCY = 2
MAX_OCCUPANCY = 8
NUM_OCCUPANCY = MAX_OCCUPANCY - MIN_OCCUPANCY + 1

ACCOUNTING = np.random.uniform(0, 10, (MAX_OCCUPANCY + 1, MAX_OCCUPANCY + 1))
PREFERENCE = np.random.randint(0, 10, (N_FAMILIES, N_DAYS + 1))

FIRST_DAY_OCCUPANCY = 4
LAST_DAY_OCCUPANCY = MIN_OCCUPANCY

INITIAL_ASSIGN = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
#random.shuffle(INITIAL_ASSIGN)

INITIAL_OCCUPACY = np.zeros(N_DAYS, dtype=np.int64)
for i, d in enumerate(INITIAL_ASSIGN):
    INITIAL_OCCUPACY[d] += FAMILY_SIZE[i]

print(INITIAL_OCCUPACY, FAMILY_SIZE)
#df = solveSantaLP(INITIAL_ASSIGN, list(range(N_DAYS)), INITIAL_OCCUPACY, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
#df = solveSantaLP(INITIAL_ASSIGN, list(range(N_FAMILIES)), INITIAL_OCCUPACY, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
#df = solveSantaLP(INITIAL_ASSIGN, list(range(N_FAMILIES)), INITIAL_OCCUPACY, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
_DESIRED = np.array([0, 1, 3, 4], dtype=np.int64).reshape(1, -1).repeat(N_FAMILIES, axis=0)
target_families = [0, 1, 2, 3, 8, 9]
DESIRED = {}
prediction = {}
for fam_id in target_families:
    DESIRED[fam_id] = list(_DESIRED[fam_id])
    prediction[fam_id] = INITIAL_ASSIGN[fam_id]

print(DESIRED)
#df = solveSantaIP(DESIRED, list(range(N_FAMILIES)), INITIAL_OCCUPACY, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
df = solveSantaIP(prediction, DESIRED, INITIAL_OCCUPACY, 4, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
#df = solveSantaLP(DESIRED, FAMILY_SIZE, PREFERENCE, ACCOUNTING)
print(df)
