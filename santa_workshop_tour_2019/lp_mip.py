import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from .const import N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY
from .cost import create_penalty_memo, create_accounting_memo


def solveSantaLP(DESIRED, family_size, penalty_memo):

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
        S.Sum([x[i, j] * family_size[i] for i in candidates[j]]) for j in range(N_DAYS)
    ]

    family_presence = [
        S.Sum([x[i, j] for j in DESIRED[i, :]]) for i in range(N_FAMILIES)
    ]

    # Objective
    preference_cost = S.Sum(
        [
            penalty_memo[i, j + 1] * x[i, j]
            for i in range(N_FAMILIES)
            for j in DESIRED[i, :]
        ]
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


def solveSantaIP(
    DESIRED, families, family_size, penalty_memo, min_occupancy, max_occupancy
):

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
        S.Sum([x[i, j] * family_size[i] for i in candidates[j]]) for j in range(N_DAYS)
    ]

    family_presence = [S.Sum([x[i, j] for j in DESIRED[i, :]]) for i in families]

    # Objective
    preference_cost = S.Sum(
        [penalty_memo[i, j + 1] * x[i, j] for i in families for j in DESIRED[i, :]]
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


def build_lp_mip(data):
    family_size = data.n_people.values
    DESIRED = data.values[:, :-1] - 1
    penalty_memo = create_penalty_memo(data)

    def solveSanta():
        df = solveSantaLP(
            DESIRED, family_size, penalty_memo
        )  # Initial solution for most of families
        THRS = 0.999

        assigned_df = df[df.n > THRS].copy()
        unassigned_df = df[(df.n <= THRS) & (df.n > 1 - THRS)]
        unassigned = unassigned_df.family_id.unique()
        print("{} unassigned families".format(len(unassigned)))

        assigned_df["family_size"] = family_size[assigned_df.family_id]
        occupancy = assigned_df.groupby("day").family_size.sum().values
        min_occupancy = np.array([max(0, MIN_OCCUPANCY - o) for o in occupancy])
        max_occupancy = np.array([MAX_OCCUPANCY - o for o in occupancy])

        rdf = solveSantaIP(
            DESIRED, unassigned, family_size, penalty_memo, min_occupancy, max_occupancy
        )  # solve the rest with MIP
        df = pd.concat((assigned_df[["family_id", "day"]], rdf)).sort_values(
            "family_id"
        )
        return df.day.values + 1

    return solveSanta
