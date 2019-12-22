import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from .const import N_DAYS, N_FAMILIES, MAX_OCCUPANCY, MIN_OCCUPANCY
from .cost import create_penalty_memo, create_accounting_memo
from .util import group_by_day
import random

def _add_delta_occupancy_constraint(S, occupancy, delta):
    for j in range(len(occupancy) - 1):
        S.Add(occupancy[j] - occupancy[j + 1] <= delta)
        S.Add(occupancy[j + 1] - occupancy[j] <= delta)


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
    for d in candidates.keys():
        vs = (
            range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            if (d + 1) in candidates
            else [occupancy[d + 1]]
        )

        for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            if d == (N_DAYS - 1):
                vs = [u]
            for v in vs:
                if accounting_memo[u, v] <= th:
                    y.setdefault(d, {}).update(
                        {(u, v): S.BoolVar("y[%i,%i,%i]" % (d, u, v))}
                    )
    return y


def _calc_family_presence(S, x, DESIRED):
    return [S.Sum([x[i, j] for j in days]) for i, days in DESIRED.items()]


def _calc_preference_cost(S, x, DESIRED, penalty_memo):
    return S.Sum(
        [penalty_memo[i, j + 1] * x[i, j] for i, days in DESIRED.items() for j in days]
    )


def _calc_accounting_cost(S, y, occupancy, accounting_memo):
    days = set(y.keys())
    ret = []
    for d, vals in y.items():
        for (u, v), yv in vals.items():
            coef = accounting_memo[u, v]
            if (0 < d) and (d - 1) not in days:
                coef += accounting_memo[occupancy[d - 1], u]
            ret.append(coef * yv)
    return S.Sum(ret)


def _add_family_presence_constraint(S, x, DESIRED):
    family_presence = _calc_family_presence(S, x, DESIRED)
    for i in range(len(family_presence)):
        S.Add(family_presence[i] == 1)


def _add_occupancy_constraint(S, candidates, occupancy):
    for j in sorted(candidates.keys()):
        S.Add(occupancy[j] >= MIN_OCCUPANCY)
        S.Add(occupancy[j] <= MAX_OCCUPANCY)


def _add_accounting_constraint(S, y, occupancy):
    for d, yd in y.items():
        y_sum_u = S.Sum([yv * u for (u, _), yv in yd.items()])
        S.Add(y_sum_u == occupancy[d])

        y_sum_v = S.Sum([yv * v for (_, v), yv in yd.items()])
        S.Add(y_sum_v == occupancy[d + 1])

        y_sum = S.Sum(yd.values())
        S.Add(y_sum == 1)


def _add_smoothness_constraint(S, y, candidates):
    for d in candidates.keys():
        if (d + 1) not in candidates:
            continue
        for t in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            y_sum_u = S.Sum(
                [
                    y[d][u, t]
                    for u in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
                    if (u, t) in y[d]
                ]
            )
            y_sum_v = S.Sum(
                [
                    y[d + 1][t, v]
                    for v in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
                    if (t, v) in y[d + 1]
                ]
            )
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


def _get_result_df(DESIRED, x):
    l = [
        (i, j, x[i, j].solution_value())
        for i, days in DESIRED.items()
        for j in days
        if x[i, j].solution_value() > 0
    ]
    return pd.DataFrame(l, columns=["family_id", "day", "n"])


def solve(
    solver,
    prediction,
    DESIRED,
    daily_occupancy,
    th,
    th2,
    family_size,
    penalty_memo,
    accounting_memo
):

    S = pywraplp.Solver(
        "SolveAssignmentProblem", solver
    )

    x, candidates = _add_penalty_candidates(S, DESIRED)
    occupancy = _calc_occupancy(S, x, prediction, daily_occupancy, candidates, family_size)
    y = _add_accounting_candidates(S, candidates, occupancy, th, accounting_memo)

    # Objective
    total_cost = _calc_preference_cost(S, x, DESIRED, penalty_memo)
    if 0 < len(y):
        total_cost += _calc_accounting_cost(
            S, y, occupancy, accounting_memo
        )
    S.Minimize(total_cost)

    # Constraints
    if 0 < len(y):
        _add_accounting_constraint(S, y, occupancy)
        _add_smoothness_constraint(S, y, candidates)
    else:
        _add_delta_occupancy_constraint(S, occupancy, th2)
    _add_family_presence_constraint(S, x, DESIRED)
    _add_occupancy_constraint(S, candidates, occupancy)

    S.Solve()
    return _get_result_df(DESIRED, x)


def build_mip(data, choices=-1, accounting_thresh=4096):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()
    DESIRED = {i: data.values[i, :choices] - 1 for i in range(data.shape[0])}

    def _mip(prediction, daily_occupancy):
        new = prediction.copy()
        daily_occupancy = daily_occupancy.copy()

        rem_days = set(range(1, 101))
        sel_days = []
        while 0 < len(rem_days):
            focus_day = random.choice(tuple(rem_days))

            sel_days.append(focus_day)
            rem_days.remove(focus_day)
            if focus_day - 1 in rem_days:
                rem_days.remove(focus_day - 1)
            if focus_day + 1 in rem_days:
                rem_days.remove(focus_day + 1)

        families_per_day = group_by_day(prediction)
        random.shuffle(sel_days)
        fam_ids = []
        for d in sel_days:
            fam_ids = [*fam_ids, *families_per_day[d]]

        unassigned = {
            i: list(set([new[i] - 1, *[d for d in DESIRED[i] if (d + 1) in sel_days]]))
            for i in fam_ids
        }
        prediction = {i: new[i] - 1 for i in fam_ids}

        df = solve(
            pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING,
            prediction,
            unassigned,
            daily_occupancy[1:],
            accounting_thresh,
            1e12,
            family_size,
            penalty_memo,
            accounting_memo
        )
        for _, row in df.iterrows():
            fam_id = int(row["family_id"])
            day = int(row["day"]) + 1
            new[fam_id] = day
        return new

    return _mip


def build_init_solver(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()

    def solveSanta(choices=7, accounting_thresh=4096):
        DESIRED = {i: data.values[i, :choices] - 1 for i in range(data.shape[0])}
        df = solve(
                pywraplp.Solver.GLOP_LINEAR_PROGRAMMING, {}, DESIRED, [0] * N_DAYS, 4096, 23, family_size, penalty_memo, accounting_memo
        )
        THRS = 0.999

        assigned_df = df[df.n > THRS].copy()
        unassigned_df = df[(df.n <= THRS) & (df.n > 1 - THRS)]
        unassigned = {i: DESIRED[i] for i in unassigned_df.family_id.unique()}
        predictions = {i: None for i in unassigned.keys()}

        assigned_df["family_size"] = family_size[assigned_df.family_id]
        occupancy = assigned_df.groupby("day").family_size.sum().values

        rdf = solve(
            pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING,
            predictions,
            unassigned,
            occupancy,
            accounting_thresh,
            1e12,
            family_size,
            penalty_memo,
            accounting_memo,
        )
        df = pd.concat((assigned_df, rdf)).sort_values("family_id")
        return df.day.values + 1

    return solveSanta
