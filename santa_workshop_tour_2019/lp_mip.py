import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from .const import N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY
from .cost import create_penalty_memo, create_accounting_memo

#N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY = 5, 10, 8, 2

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

    for i, days in DESIRED.items():
        for j in days:
            candidates[j].append(i)
            x[i, j] = S.BoolVar("x[%i,%i]" % (i, j))

    daily_occupancy = [
        S.Sum([x[i, j] * family_size[i] for i in candidates[j]]) for j in range(N_DAYS)
    ]

    family_presence = [
        S.Sum([x[i, j] for j in days]) for i, days in DESIRED.items()
    ]

    # Objective
    preference_cost = S.Sum(
        [
            penalty_memo[i, j + 1] * x[i, j]
            for i, days in DESIRED.items()
            for j in days
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
        for i, days in DESIRED.items()
        for j in days
        if x[i, j].solution_value() > 0
    ]

    df = pd.DataFrame(l, columns=["family_id", "day", "n"])
    return df


def _add_penalty_candidates(S, DESIRED):
    x = {}
    candidates = [
        [] for _ in range(N_DAYS)
    ]  # families that can be assigned to each day

    for i, days in DESIRED.items():
        for j in days:
            candidates[j].append(i)
            x[i, j] = S.BoolVar("x[%i,%i]" % (i, j))
    return x, candidates

def _add_accounting_candidates(S, candidates, th, accounting_memo):
    y = {}
    for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
        for k in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            if th < accounting_memo[j, k]:
                continue
            for i, cs in enumerate(candidates):
                if len(cs) == 0:
                    continue
                if i not in y:
                    y[i] = {}
                y[i][j, k] = S.BoolVar("y[%i,%i,%i]" % (i, j, k))
    return y


def _calc_faily_presence(S, x, DESIRED):
    return [S.Sum([x[i, j] for j in days]) for i, days in DESIRED.items()]


def _calc_preference_cost(S, x, DESIRED, penalty_memo):
    return S.Sum(
        [penalty_memo[i, j + 1] * x[i, j] for i, days in DESIRED.items() for j in days]
    )

def _calc_accounting_cost(S, y, accounting_memo):
    days = set(y.keys())
    return S.Sum(
        [
            accounting_memo[u, v] * yv
            for d in days for (u, v), yv in y[d].items()
        ]
    )

def _add_family_presence_constraint(S, x, DESIRED):
    family_presence = _calc_faily_presence(S, x, DESIRED)
    for i in range(len(family_presence)):
        S.Add(family_presence[i] == 1)

def _add_occupancy_constraint(S, occupancy):
    for j in range(N_DAYS):
        S.Add(occupancy[j] >= MIN_OCCUPANCY)
        S.Add(occupancy[j] <= MAX_OCCUPANCY)


def _add_accounting_constraint(S, y, occupancy):
    for d, yd in y.items():
        y_sum_u = S.Sum(
            [
                yv * u for (u, _), yv in yd.items()
            ]
        )
        S.Add(y_sum_u == occupancy[d])

        y_sum_v = S.Sum(
            [
                yv * v for (_, v), yv in yd.items()
            ]
        )
        S.Add(y_sum_v == occupancy[d + 1])

        y_sum = S.Sum(yd.values())
        S.Add(y_sum == 1)


def _add_smoothness_constraint(S, y, candidates):
#    for d, yv in y.items():
#        if len(candidates) - 1 <= d:
#            continue
#        us = {}
#        vs = {}
#        for u, v in yv.keys():
#            us.setdefault(u, set()).add(v)
#            vs.setdefault(v, set()).add(u)
#        for t, uu in vs.items():
#            y_sum_u = S.Sum([yv[u, t] for u in uu])
#            if 0 < len(candidates[(d + 1)]):
#                y_sum_v = S.Sum([y[d + 1][t, v] for v in us[t]])
#                S.Add(y_sum_u == y_sum_v)
    for d, cs in enumerate(candidates[:-1]):
        if len(cs) == 0:
            continue
        for t in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            y_sum_u = S.Sum([y[d][u, t] for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) if (u,t) in y[d]])
            if 0 < len(candidates[(d + 1)]):
                y_sum_v = S.Sum([y[d + 1][t, v] for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) if (t,v) in y[d+1]])
                S.Add(y_sum_u == y_sum_v)


def _calc_occupancy(S, x, prediction, occupancy, candidates, family_size):
    occupancy = list(occupancy.copy())
    for fam_id, d in prediction.items():
        if d is not None:
            occupancy[d] -= family_size[fam_id]
    for d in range(N_DAYS):
        occupancy[d] += S.Sum([x[i, d] * family_size[i] for i in candidates[d]])
    occupancy.append(occupancy[-1])
    return occupancy


def solveSantaIP(
    prediction, DESIRED, daily_occupancy, th, family_size, penalty_memo, accounting_memo
):

    S = pywraplp.Solver(
        "SolveAssignmentProblem", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
    )

    # S.SetNumThreads(NumThreads)
    # S.set_time_limit(limit_in_seconds*1000*NumThreads) #cpu time = wall time * N_threads

    x, candidates = _add_penalty_candidates(S, DESIRED)
    y = _add_accounting_candidates(S, candidates, th, accounting_memo)
    occupancy = _calc_occupancy(S, x, prediction, daily_occupancy, candidates, family_size)

    # Objective
    preference_cost = _calc_preference_cost(S, x, DESIRED, penalty_memo)
    accounting_cost = _calc_accounting_cost(S, y, accounting_memo)
    S.Minimize(preference_cost + accounting_cost)

    # Constraints
    _add_family_presence_constraint(S, x, DESIRED)
    _add_occupancy_constraint(S, occupancy)
    _add_accounting_constraint(S, y, occupancy)
    _add_smoothness_constraint(S, y, candidates)

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
        (i, j) for i, days in DESIRED.items() for j in days if x[i, j].solution_value() > 0
    ]

    df = pd.DataFrame(l, columns=["family_id", "day"])
    return df


def build_lp_mip(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()
    DESIRED = {i: data.values[i, :-1] -1 for i in range(data.shape[0])}

    def solveSanta():
        df = solveSantaLP(
            DESIRED, family_size, penalty_memo
        )  # Initial solution for most of families
        THRS = 0.999

        assigned_df = df[df.n > THRS].copy()
        unassigned_df = df[(df.n <= THRS) & (df.n > 1 - THRS)]
        unassigned = {i: DESIRED[i] for i in unassigned_df.family_id.unique()}
        predictions = {i: None for i in unassigned.keys()}
        print("{} unassigned families".format(len(unassigned)))

        assigned_df["family_size"] = family_size[assigned_df.family_id]
        occupancy = assigned_df.groupby("day").family_size.sum().values
        min_occupancy = np.array([max(0, MIN_OCCUPANCY - o) for o in occupancy])
        max_occupancy = np.array([MAX_OCCUPANCY - o for o in occupancy])

        rdf = solveSantaIP(
            predictions, DESIRED, occupancy, 4096, family_size, penalty_memo, accounting_memo
        )  # solve the rest with MIP
        df = pd.concat((assigned_df[["family_id", "day"]], rdf)).sort_values(
            "family_id"
        )
        return df.day.values + 1

    return solveSanta
