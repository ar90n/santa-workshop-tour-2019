import numpy as np
import pandas as pd
from numba import njit
from itertools import product
from ortools.linear_solver import pywraplp
from .io import load_data


def get_penalty(n, choice):
    penalty = None
    if choice == 0:
        penalty = 0
    elif choice == 1:
        penalty = 50
    elif choice == 2:
        penalty = 50 + 9 * n
    elif choice == 3:
        penalty = 100 + 9 * n
    elif choice == 4:
        penalty = 200 + 9 * n
    elif choice == 5:
        penalty = 200 + 18 * n
    elif choice == 6:
        penalty = 300 + 18 * n
    elif choice == 7:
        penalty = 300 + 36 * n
    elif choice == 8:
        penalty = 400 + 36 * n
    elif choice == 9:
        penalty = 500 + 36 * n + 199 * n
    else:
        penalty = 500 + 36 * n + 398 * n
    return penalty


def GetPreferenceCostMatrix(data):
    cost_matrix = np.zeros((N_FAMILIES, N_DAYS), dtype=np.int64)
    for i in range(N_FAMILIES):
        desired = data.values[i, :-1]
        cost_matrix[i, :] = get_penalty(FAMILY_SIZE[i], 10)
        for j, day in enumerate(desired):
            cost_matrix[i, day - 1] = get_penalty(FAMILY_SIZE[i], j)
    return cost_matrix


def GetAccountingCostMatrix():
    ac = np.zeros((1000, 1000), dtype=np.float64)
    for n in range(ac.shape[0]):
        for n_p1 in range(ac.shape[1]):
            diff = abs(n - n_p1)
            ac[n, n_p1] = max(0, (n - 125) / 400 * n ** (0.5 + diff / 50.0))
    return ac


# preference cost
@njit(fastmath=True)
def pcost(prediction):
    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)
    penalty = 0
    for (i, p) in enumerate(prediction):
        n = FAMILY_SIZE[i]
        penalty += PCOSTM[i, p]
        daily_occupancy[p] += n
    return penalty, daily_occupancy


# accounting cost
@njit(fastmath=True)
def acost(daily_occupancy):
    accounting_cost = 0
    n_out_of_range = 0
    daily_occupancy[-1] = daily_occupancy[-2]
    for day in range(N_DAYS):
        n_p1 = daily_occupancy[day + 1]
        n = daily_occupancy[day]
        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)
        accounting_cost += ACOSTM[n, n_p1]
    return accounting_cost, n_out_of_range


@njit(fastmath=True)
def cost_function(prediction):
    penalty, daily_occupancy = pcost(prediction)
    accounting_cost, n_out_of_range = acost(daily_occupancy)
    return penalty + accounting_cost + n_out_of_range * 100000000


N_DAYS = 100
N_FAMILIES = 5000
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

data = load_data()

FAMILY_SIZE = data.n_people.values
DESIRED = data.values[:, :-1] - 1
PCOSTM = GetPreferenceCostMatrix(data)  # Preference cost matrix
ACOSTM = GetAccountingCostMatrix()  # Accounting cost matrix


def solveSantaLP():

    S = pywraplp.Solver(
        "SolveAssignmentProblem", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
    )

    # S.SetNumThreads(NumThreads)
    # S.set_time_limit(limit_in_seconds*1000*NumThreads) #cpu time = wall time * N_threads

    x = {}
    candidates = [
        [] for _ in range(N_DAYS)
    ]  # families that can be assigned to each day

    for i in range(N_FAMILIES):
        for j in DESIRED[i, :]:
            candidates[j].append(i)
            x[i, j] = S.BoolVar("x[%i,%i]" % (i, j))

    daily_occupancy = [
        S.Sum([x[i, j] * FAMILY_SIZE[i] for i in candidates[j]]) for j in range(N_DAYS)
    ]

    family_presence = [
        S.Sum([x[i, j] for j in DESIRED[i, :]]) for i in range(N_FAMILIES)
    ]

    # Objective
    preference_cost = S.Sum(
        [PCOSTM[i, j] * x[i, j] for i in range(N_FAMILIES) for j in DESIRED[i, :]]
    )

    S.Minimize(preference_cost)

    # Constraints
    for j in range(N_DAYS - 1):
        S.Add(daily_occupancy[j] - daily_occupancy[j + 1] <= 23)
        S.Add(daily_occupancy[j + 1] - daily_occupancy[j] <= 23)

    for i in range(N_FAMILIES):
        S.Add(family_presence[i] == 1)

    for j in range(N_DAYS):
        S.Add(daily_occupancy[j] >= MIN_OCCUPANCY)
        S.Add(daily_occupancy[j] <= MAX_OCCUPANCY)

    res = S.Solve()

    resdict = {
        0: "OPTIMAL",
        1: "FEASIBLE",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "ABNORMAL",
        5: "MODEL_INVALID",
        6: "NOT_SOLVED",
    }

    print("LP solver result:", resdict[res])

    l = [
        (i, j, x[i, j].solution_value())
        for i in range(N_FAMILIES)
        for j in DESIRED[i, :]
        if x[i, j].solution_value() > 0
    ]

    df = pd.DataFrame(l, columns=["family_id", "day", "n"])
    return df


def solveSantaIP(families, min_occupancy, max_occupancy):

    S = pywraplp.Solver(
        "SolveAssignmentProblem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    # S.SetNumThreads(NumThreads)
    # S.set_time_limit(limit_in_seconds*1000*NumThreads) #cpu time = wall time * N_threads

    n_families = len(families)

    x = {}
    candidates = [
        [] for _ in range(N_DAYS)
    ]  # families that can be assigned to each day

    for i in families:
        for j in DESIRED[i, :]:
            candidates[j].append(i)
            x[i, j] = S.BoolVar("x[%i,%i]" % (i, j))

    daily_occupancy = [
        S.Sum([x[i, j] * FAMILY_SIZE[i] for i in candidates[j]]) for j in range(N_DAYS)
    ]

    family_presence = [S.Sum([x[i, j] for j in DESIRED[i, :]]) for i in families]

    # Objective
    preference_cost = S.Sum(
        [PCOSTM[i, j] * x[i, j] for i in families for j in DESIRED[i, :]]
    )

    S.Minimize(preference_cost)

    # Constraints

    for i in range(n_families):
        S.Add(family_presence[i] == 1)

    for j in range(N_DAYS):
        S.Add(daily_occupancy[j] >= min_occupancy[j])
        S.Add(daily_occupancy[j] <= max_occupancy[j])

    res = S.Solve()

    resdict = {
        0: "OPTIMAL",
        1: "FEASIBLE",
        2: "INFEASIBLE",
        3: "UNBOUNDED",
        4: "ABNORMAL",
        5: "MODEL_INVALID",
        6: "NOT_SOLVED",
    }

    print("MIP solver result:", resdict[res])

    l = [
        (i, j) for i in families for j in DESIRED[i, :] if x[i, j].solution_value() > 0
    ]

    df = pd.DataFrame(l, columns=["family_id", "day"])
    return df


def solveSanta():
    df = solveSantaLP()  # Initial solution for most of families
    THRS = 0.999

    assigned_df = df[df.n > THRS].copy()
    unassigned_df = df[(df.n <= THRS) & (df.n > 1 - THRS)]
    unassigned = unassigned_df.family_id.unique()
    print("{} unassigned families".format(len(unassigned)))

    assigned_df["family_size"] = FAMILY_SIZE[assigned_df.family_id]
    occupancy = assigned_df.groupby("day").family_size.sum().values
    min_occupancy = np.array([max(0, MIN_OCCUPANCY - o) for o in occupancy])
    max_occupancy = np.array([MAX_OCCUPANCY - o for o in occupancy])

    rdf = solveSantaIP(
        unassigned, min_occupancy, max_occupancy
    )  # solve the rest with MIP
    df = pd.concat((assigned_df[["family_id", "day"]], rdf)).sort_values("family_id")
    return df.day.values + 1
