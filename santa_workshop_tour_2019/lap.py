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

def solve(data, occupancies):
    weights = _get_weights(data, occupancies)
    least_cost, col, row = lapjv(weights)

    return _calc_days(occupancies, col, weights.shape[0])


def _get_weights(data, occupancies):
    weights = create_penalty_memo(data)
    weights = weights[:, 1:].reshape(5000, 100)
    weights = weights.repeat(occupancies[1:], axis=1)
    weights = weights.reshape(5000, 5000)

    return weights
