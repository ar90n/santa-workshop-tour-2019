"""
Reference:
    - https://www.kaggle.com/xhlulu/santa-s-2019-faster-cost-function-24-s
"""

from .const import N_DAYS, MAX_OCCUPANCY

from functools import partial, lru_cache

from numba import njit
import numpy as np
import pandas as pd


def create_accounting_memo():
    accounting_matrix = np.zeros((MAX_OCCUPANCY + 1, MAX_OCCUPANCY + 1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(MAX_OCCUPANCY + 1):
        for diff in range(MAX_OCCUPANCY + 1):
            accounting_cost = (
                (today_count - 125.0) / 400.0 * today_count ** (0.5 + diff / 50.0)
            )
            accounting_matrix[today_count, diff] = max(0, accounting_cost)

    return accounting_matrix


def create_penalty_memo(data):
    family_size = data.n_people.values
    penalties_array = np.array(
        [
            [
                0,
                50,
                50 + 9 * n,
                100 + 9 * n,
                200 + 9 * n,
                200 + 18 * n,
                300 + 18 * n,
                300 + 36 * n,
                400 + 36 * n,
                500 + 36 * n + 199 * n,
                500 + 36 * n + 398 * n,
            ]
            for n in range(family_size.max() + 1)
        ]
    )

    choice_matrix = data.loc[:, "choice_0":"choice_9"].values

    N = family_size.shape[0]
    penalty_matrix = penalties_array[family_size[range(N)], -1].reshape(
        -1, 1
    ) * np.ones(N_DAYS + 1)
    for i in range(N):
        penalty_matrix[i, choice_matrix[i]] = penalties_array[family_size[i], :-1]

    return penalty_matrix


@njit
def _compute_cost_fast(
    prediction,
    family_size,
    days_array,
    penalty_memo,
    accounting_memo,
    max_occupancy,
    min_occupancy,
):
    """
    Do not use this function. Please use `build_cost_function` instead to
    build your own "cost_function".
    """
    N = family_size.shape[0]
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros(len(days_array) + 1, dtype=np.int64)
    penalty = 0

    # Looping over each family; d is the day, n is size of that family
    for i in range(N):
        n = family_size[i]
        d = prediction[i]

        daily_occupancy[d] += n
        penalty += penalty_memo[i, d]

    # for each date, check total occupancy
    # (using soft constraints instead of hard constraints)
    # Day 0 does not exist, so we do not count it
    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > max_occupancy) | (relevant_occupancy < min_occupancy)
    )

    if incorrect_occupancy:
        return 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    init_occupancy = daily_occupancy[days_array[0]]
    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy ** (0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)

    # Loop over the rest of the days_array, keeping track of previous count
    yesterday_count = init_occupancy
    for day in days_array[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += accounting_memo[today_count, diff]
        yesterday_count = today_count

    total_cost = penalty + accounting_cost
    return total_cost


def build_cost_function(data, max_occupancy=300, min_occupancy=125):
    """
    data (pd.DataFrame):
        should be the df that contains family information. Preferably load it from "family_data.csv".
    """
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)

    # Precompute matrices needed for our cost function
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()

    # Partially apply `_compute_cost_fast` so that the resulting partially applied
    # function only requires prediction as input. E.g.
    # Non partial applied: score = _compute_cost_fast(prediction, family_size, days_array, ...)
    # Partially applied: score = cost_function(prediction)
    cost_function = partial(
        _compute_cost_fast,
        family_size=family_size,
        days_array=days_array,
        penalty_memo=penalty_memo,
        accounting_memo=accounting_memo,
        max_occupancy=max_occupancy,
        min_occupancy=min_occupancy,
    )

    return cost_function
