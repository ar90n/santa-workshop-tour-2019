from lap import lapjv
import numpy as np
from .cost import create_penalty_memo, create_accounting_memo, _partial_accounting_cost
from numba import njit
import random


def group_by_size(family_size):
    fams = {}
    for fam_id in range(len(family_size)):
        fam_size = family_size[fam_id]
        fams.setdefault(fam_size, []).append(fam_id)
    return {k: np.array(v, dtype=np.int64) for k, v in fams.items()}


def group_by_day(prediction):
    fams = {}
    for fam_id in range(len(prediction)):
        day = prediction[fam_id]
        fams.setdefault(day, []).append(fam_id)
    return {k: np.array(v, dtype=np.int64) for k, v in fams.items()}


@njit
def _calc_days(occupancies, col, l):
    acc = np.cumsum(occupancies)
    best = np.zeros(l, dtype=np.int64)
    for i in range(len(col)):
        c = col[i]
        j = 1
        while acc[j] <= c:
            j += 1
        best[i] += j
    return best


@njit
def _create_weights(prediction, fam_ids, penalty_memo):
    weights = np.zeros((len(fam_ids), len(fam_ids)), dtype=np.int64)
    for i in range(len(fam_ids)):
        for j in range(len(fam_ids)):
            if i == j:
                continue
            idi = fam_ids[i]
            idj = fam_ids[j]
            di = prediction[idi]
            dj = prediction[idj]
            weights[i, j] = penalty_memo[idi, dj] - penalty_memo[idi, di]
    return weights


def build_family_lap(data):
    family_size = data.n_people.values
    weights_source = create_penalty_memo(data)[:, 1:].reshape(5000, 100)

    def _family_lap(slots):
        weights = weights_source.repeat(slots[1:], axis=1)
        _, col, _ = lapjv(weights)
        best = _calc_days(slots, col, weights.shape[0])

        daily_occupancy = np.zeros(101, dtype=np.int64)
        for j in range(len(best)):
            daily_occupancy[best[j]] += family_size[j]

        return best, daily_occupancy

    return _family_lap


def build_family_size_lap(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)
    fams = group_by_size(family_size)

    def _family_size_lap(prediction):
        new = prediction.copy()
        for fam_ids in fams.values():
            weights = _create_weights(prediction, fam_ids, penalty_memo)
            _, col, _ = lapjv(weights)

            for i in range(len(col)):
                new[fam_ids[i]] = prediction[fam_ids[col[i]]]
        return new

    return _family_size_lap


def build_non_adj_family_lap(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()

    def _non_adj_family_lap(prediction, daily_occupancy):
        families_per_day = group_by_day(prediction)
        new = prediction.copy()
        daily_occupancy = daily_occupancy.copy()

        days = set(range(1, 101))
        fam_ids = []
        while 0 < len(days):
            focus_day = random.choice(tuple(days))
            fam_id = random.choice(families_per_day[focus_day])
            fam_ids.append(fam_id)

            days.remove(focus_day)
            if focus_day - 1 in days:
                days.remove(focus_day - 1)
            if focus_day + 1 in days:
                days.remove(focus_day + 1)

        weights = np.zeros((len(fam_ids), len(fam_ids)), dtype=np.int64)
        for i in range(len(fam_ids)):
            for j in range(len(fam_ids)):
                idi = fam_ids[i]
                idj = fam_ids[j]
                di = prediction[idi]
                dj = prediction[idj]
                diff_size = family_size[idi] - family_size[idj]
                weights[i, j] = (
                    penalty_memo[idi, dj]
                    + _partial_accounting_cost(
                        daily_occupancy, dj, diff_size, 0, accounting_memo
                    )
                    + _partial_accounting_cost(
                        daily_occupancy, dj - 1, 0, diff_size, accounting_memo
                    )
                )

        _, col, _ = lapjv(weights)
        for i in range(len(col)):
            daily_occupancy[new[fam_ids[i]]] -= family_size[fam_ids[i]]
            daily_occupancy[prediction[fam_ids[col[i]]]] += family_size[fam_ids[i]]
            new[fam_ids[i]] = prediction[fam_ids[col[i]]]
        return new, daily_occupancy

    return _non_adj_family_lap
