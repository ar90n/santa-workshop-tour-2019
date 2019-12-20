import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from .const import N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY
from .cost import create_penalty_memo, create_accounting_memo

#N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY = 5, 10, 8, 2

def solveSantaLP(DESIRED, family_size, penalty_memo, accounting_memo):

    S = pywraplp.Solver(
        "SolveAssignmentProblem", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING
    )

    # S.SetNumThreads(NumThreads)
    # S.set_time_limit(limit_in_seconds*1000*NumThreads) #cpu time = wall time * N_threads

    x, candidates = _add_penalty_candidates(S, DESIRED)
    occupancy = _calc_occupancy(S, x, {}, [0] * len(family_size), candidates, family_size)

    # 
    preference_cost = _calc_preference_cost(S, x, DESIRED, penalty_memo)
    S.Minimize(preference_cost)

    # Constraints
    for j in range(N_DAYS - 1):
        S.Add(occupancy[j] - occupancy[j + 1] <= 23)
        S.Add(occupancy[j + 1] - occupancy[j] <= 23)
    _add_family_presence_constraint(S, x, DESIRED)
    _add_occupancy_constraint(S, candidates, occupancy)

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
    candidates = {}

    for i, days in DESIRED.items():
        for j in days:
            candidates.setdefault(j, []).append(i)
            x[i, j] = S.BoolVar("x[%i,%i]" % (i, j))
    return x, candidates

def _add_accounting_candidates(S, candidates, occupancy, th, accounting_memo):
    y = {}
    for d, cs in candidates.items():
        vs = range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) if (d == (N_DAYS-1) or (d + 1) in candidates) else [occupancy[d+1]]
        for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            for v in vs:
                if accounting_memo[u, v] <= th:
                    y.setdefault(d, {}).update({(u, v):S.BoolVar("y[%i,%i,%i]" % (d, u, v))})
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

def _add_occupancy_constraint(S, candidates, occupancy):
    for j in sorted(candidates.keys()):
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
    for d, cs in candidates.items():
        if (d + 1) not in candidates:
            continue
        for t in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            y_sum_u = S.Sum([y[d][u, t] for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) if (u,t) in y[d]])
            y_sum_v = S.Sum([y[d + 1][t, v] for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1) if (t,v) in y[d+1]])
            S.Add(y_sum_u == y_sum_v)


def _calc_occupancy(S, x, prediction, occupancy, candidates, family_size):
    occupancy = list(occupancy.copy())
    for fam_id, d in prediction.items():
        if d is not None:
            occupancy[d] -= family_size[fam_id]
    for d, cs in candidates.items():
        occupancy[d] += S.Sum([x[i, d] * family_size[i] for i in cs])
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
    occupancy = _calc_occupancy(S, x, prediction, daily_occupancy, candidates, family_size)
    y = _add_accounting_candidates(S, candidates, occupancy, th, accounting_memo)

    # Objective
    total_cost = _calc_preference_cost(S, x, DESIRED, penalty_memo)
    if 0 < len(y):
        total_cost += _calc_accounting_cost(S, y, accounting_memo)
    S.Minimize(total_cost)

    # Constraints
    _add_family_presence_constraint(S, x, DESIRED)
    _add_occupancy_constraint(S, candidates, occupancy)
    if 0 < len(y):
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
            DESIRED, family_size, penalty_memo, accounting_memo
        )  # Initial solution for most of families
        THRS = 0.999

        assigned_df = df[df.n > THRS].copy()
        unassigned_df = df[(df.n <= THRS) & (df.n > 1 - THRS)]
        unassigned = {i: DESIRED[i] for i in unassigned_df.family_id.unique()}
        predictions = {i: None for i in unassigned.keys()}
        print("{} unassigned families".format(len(unassigned)))

        assigned_df["family_size"] = family_size[assigned_df.family_id]
        occupancy = assigned_df.groupby("day").family_size.sum().values

        rdf = solveSantaIP(
            predictions, unassigned, occupancy, 512, family_size, penalty_memo, accounting_memo
        )  # solve the rest with MIP
        df = pd.concat((assigned_df[["family_id", "day"]], rdf)).sort_values(
            "family_id"
        )
        return df.day.values + 1

    return solveSanta
