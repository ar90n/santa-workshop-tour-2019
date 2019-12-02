"""
Reference:
    - https://www.kaggle.com/xhlulu/santa-s-2019-faster-cost-function-24-s
"""
from functools import partial

from numba import njit
import numpy as np
import pandas as pd


## Intermediate Helper Functions
def _build_choice_array(data, n_days):
    choice_matrix = data.loc[:, "choice_0":"choice_9"].values
    choice_array_num = np.full((data.shape[0], n_days + 1), -1)

    for i, choice in enumerate(choice_matrix):
        for d, day in enumerate(choice):
            choice_array_num[i, day] = d

    return choice_array_num


def _precompute_accounting(max_day_count, max_diff):
    accounting_matrix = np.zeros((max_day_count + 1, max_diff + 1))
    # Start day count at 1 in order to avoid division by 0
    for today_count in range(1, max_day_count + 1):
        for diff in range(max_diff + 1):
            accounting_cost = (
                (today_count - 125.0) / 400.0 * today_count ** (0.5 + diff / 50.0)
            )
            accounting_matrix[today_count, diff] = max(0, accounting_cost)

    return accounting_matrix


def _precompute_penalties(choice_array_num, family_size):
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

    penalty_matrix = np.zeros(choice_array_num.shape)
    N = family_size.shape[0]
    for i in range(N):
        choice = choice_array_num[i]
        n = family_size[i]

        for j in range(penalty_matrix.shape[1]):
            penalty_matrix[i, j] = penalties_array[n, choice[j]]

    return penalty_matrix


@njit
def _compute_cost_fast(
    prediction,
    family_size,
    days_array,
    penalty_matrix,
    accounting_matrix,
    MAX_OCCUPANCY,
    MIN_OCCUPANCY,
    N_DAYS,
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
        penalty += penalty_matrix[i, d]

    # for each date, check total occupancy
    # (using soft constraints instead of hard constraints)
    # Day 0 does not exist, so we do not count it
    relevant_occupancy = daily_occupancy[1:]
    incorrect_occupancy = np.any(
        (relevant_occupancy > MAX_OCCUPANCY) | (relevant_occupancy < MIN_OCCUPANCY)
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
        accounting_cost += accounting_matrix[today_count, diff]
        yesterday_count = today_count

    penalty += accounting_cost

    return penalty


def build_cost_function(data, N_DAYS=100, MAX_OCCUPANCY=300, MIN_OCCUPANCY=125):
    """
    data (pd.DataFrame):
        should be the df that contains family information. Preferably load it from "family_data.csv".
    """
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)

    # Precompute matrices needed for our cost function
    choice_array_num = _build_choice_array(data, N_DAYS)
    penalty_matrix = _precompute_penalties(choice_array_num, family_size)
    accounting_matrix = _precompute_accounting(
        max_day_count=MAX_OCCUPANCY, max_diff=MAX_OCCUPANCY
    )

    # Partially apply `_compute_cost_fast` so that the resulting partially applied
    # function only requires prediction as input. E.g.
    # Non partial applied: score = _compute_cost_fast(prediction, family_size, days_array, ...)
    # Partially applied: score = cost_function(prediction)
    cost_function = partial(
        _compute_cost_fast,
        family_size=family_size,
        days_array=days_array,
        penalty_matrix=penalty_matrix,
        accounting_matrix=accounting_matrix,
        MAX_OCCUPANCY=MAX_OCCUPANCY,
        MIN_OCCUPANCY=MIN_OCCUPANCY,
        N_DAYS=N_DAYS,
    )

    return cost_function


def get_weights(data, N_DAYS=100):
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)
    choice_array_num = _build_choice_array(data, N_DAYS)
    weights = _precompute_penalties(choice_array_num, family_size)
    weights = weights[:, 1:].reshape(5000, 100, 1)
    weights = np.tile(weights, (1, 1, 50))
    weights = weights.reshape(5000, 5000)

    return weights
