from lap import lapjv
import numpy as np
from .cost import create_penalty_memo


def solve(data):
    weights = _get_weights(data)
    least_cost, col, row = lapjv(weights)
    best = col // 50 + 1
    return best


def _get_weights(data):
    weights = create_penalty_memo(data)
    weights = weights[:, 1:].reshape(5000, 100, 1)
    weights = np.tile(weights, (1, 1, 50))
    weights = weights.reshape(5000, 5000)

    return weights
