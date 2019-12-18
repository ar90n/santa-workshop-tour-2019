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
    DESIRED, families, daily_occupancy, family_size, penalty_memo, accounting_memo
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

    y = {}
    for i, cs in enumerate(candidates):
        if len(cs) == 0:
            continue
        for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            for k in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
                y[i, j, k] = S.BoolVar("y[%i,%i,%i]" % (i, j, k))

    occupancy = list(daily_occupancy.copy())
    #for fam_id in families:
    #    occupancy[prediction[fam_id]] -= family_size[fam_id]

    for d in range(N_DAYS):
        occupancy[d] += S.Sum([x[i, d] * family_size[i] for i in candidates[d]])
    occupancy.append(occupancy[-1])


    family_presence = [S.Sum([x[i, j] for j in DESIRED[i, :]]) for i in families]

    # Objective
    preference_cost = S.Sum(
        [penalty_memo[i, j + 1] * x[i, j] for i in families for j in DESIRED[i, :]]
    )

    accounting_cost = S.Sum(
        [
            accounting_memo[u, v] * y[d, u, v]
            for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            for d, cs in enumerate(candidates) if 0 < len(cs)
        ]
    )

    S.Minimize(preference_cost + accounting_cost)

    # Constraints

    for i in range(n_families):
        S.Add(family_presence[i] == 1)

    for j in range(N_DAYS):
        S.Add(occupancy[j] >= MIN_OCCUPANCY)
        S.Add(occupancy[j] <= MAX_OCCUPANCY)

    for d, cs in enumerate(candidates):
        if len(cs) == 0:
            continue
        y_sum_u = S.Sum(
            [
                y[d, u, v] * u
                for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
                for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            ]
        )
        S.Add(y_sum_u == occupancy[d])

        y_sum_v = S.Sum(
            [
                y[d, u, v] * v
                for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
                for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            ]
        )
        S.Add(y_sum_v == occupancy[d + 1])

        y_sum = S.Sum(
            [y[d, u, v] for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)]
        )
        S.Add(y_sum == 1)

    for d, cs in enumerate(candidates[:-1]):
        if len(cs) == 0:
            continue
        for t in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            y_sum_u = S.Sum([y[d, u, t] for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)])
            if 0 < len(candidates[(d + 1)]):
                y_sum_v = S.Sum([y[d + 1, t, v] for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)])
                S.Add(y_sum_u == y_sum_v)

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
    accounting_memo = create_accounting_memo()

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
            DESIRED, unassigned, occupancy, family_size, penalty_memo, accounting_memo
        )  # solve the rest with MIP
        df = pd.concat((assigned_df[["family_id", "day"]], rdf)).sort_values(
            "family_id"
        )
        return df.day.values + 1

    return solveSanta
