from lap import lapjv
import numpy as np
from .cost import create_penalty_memo
from numba import njit

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


def _get_weights(data, occupancies):
    weights = create_penalty_memo(data)
    weights = weights[:, 1:].reshape(5000, 100)
    weights = weights.repeat(occupancies[1:], axis=1)
    weights = weights.reshape(5000, 5000)

    return weights

def build_family_lap(data):
    family_size = data.n_people.values
    weights_source = create_penalty_memo(data)[:, 1:].reshape(5000, 100)

    def _family_lap(data, occupancies):
        weights = weights_source.repeat(occupancies[1:], axis=1)
        _, col, _ = lapjv(weights)
        best = _calc_days(occupancies, col, weights.shape[0])

        daily_occupancy = np.zeros(101, dtype=np.int64)
        for j in range(len(best):
            daily_occupancy[best[j]] += family_size[j]

        return best, daily_occupancy

    return _family_lap


def build_family_size_lap(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)

    fams = {}
    for fam_id in range(len(family_size)):
        fam_size = family_size[fam_id]
        fams.setdefault(fam_size, []).append(fam_id)

    @njit
    def _create_weights(prediction, fam_ids):
        weights = np.zeros((len(fam_ids), len(fam_ids)), dtype=np.int64)
        for i in range(len(fam_ids)):
            for j in range(len(fam_ids)):
                if i == j:
                    continue
                idi = fam_ids[i]
                idj = fam_ids[j]
                di = prediction[idi]
                dj = prediction[idj]
                weights[i, j] = (penalty_memo[idi, dj] - penalty_memo[idi, di])
        return weights

    def _family_size_lap(prediction):
        new = prediction.copy()
        for fam_ids in fams.values():
            weights = _create_weights(prediction, fam_ids)
            _, col, _ = lapjv(weights)

            for i in range(len(col)):
                new[fam_ids[i]] = prediction[fam_ids[col[i]]]
        return new
    return _family_size_lap
