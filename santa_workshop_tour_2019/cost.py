"""
Reference:
    - https://www.kaggle.com/xhlulu/santa-s-2019-faster-cost-function-24-s
"""

from .const import N_DAYS, MAX_OCCUPANCY

from numba import njit
import numpy as np


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
def _acc_prediction(prediction, family_size, penalty_memo):
    N = family_size.shape[0]
    days_array = np.arange(N_DAYS, 0, -1)
    # We'll use this to count the number of people scheduled each day
    daily_occupancy = np.zeros(len(days_array) + 1, dtype=np.int64)
    penalty = 0

    # Looping over each family; d is the day, n is size of that family
    for i in range(N):
        n = family_size[i]
        d = prediction[i]

        daily_occupancy[d] += n
        penalty += penalty_memo[i, d]
    return daily_occupancy, penalty


@njit
def _total_cost(
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
    daily_occupancy, penalty = _acc_prediction(prediction, family_size, penalty_memo)

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


@njit
def _partial_accounting_cost(
    daily_occupancy, today, diff_today, diff_yesterday, accounting_memo
):
    if today <= 0:
        return 0

    today_count = daily_occupancy[today] + diff_today
    if (today_count < 125) or (300 < today_count):
        return 100000000

    if (today + 1) < len(daily_occupancy):
        yesterday_count = daily_occupancy[today + 1] + diff_yesterday
    else:
        yesterday_count = today_count
    diff_count = abs(today_count - yesterday_count)
    return accounting_memo[today_count, diff_count]


@njit
def _delta_swap_cost(daily_occupancy, d1, d2, c1, c2, penalty_memo, accounting_memo):
    if d1 < d2:
        major_day = d2
        minor_day = d1
        major_diff = c1 - c2
        minor_diff = c2 - c1
    else:
        major_day = d1
        minor_day = d2
        major_diff = c2 - c1
        minor_diff = c1 - c2

    new = 0
    old = 0

    new += _partial_accounting_cost(
        daily_occupancy, minor_day - 1, 0, minor_diff, accounting_memo
    )
    old += _partial_accounting_cost(
        daily_occupancy, minor_day - 1, 0, 0, accounting_memo
    )

    new += _partial_accounting_cost(
        daily_occupancy, major_day, major_diff, 0, accounting_memo
    )
    old += _partial_accounting_cost(daily_occupancy, major_day, 0, 0, accounting_memo)

    if (minor_day + 1) == major_day:
        new += _partial_accounting_cost(
            daily_occupancy, minor_day, minor_diff, major_diff, accounting_memo
        )
        old += _partial_accounting_cost(
            daily_occupancy, minor_day, 0, 0, accounting_memo
        )
    else:
        new += _partial_accounting_cost(
            daily_occupancy, minor_day, minor_diff, 0, accounting_memo
        )
        old += _partial_accounting_cost(
            daily_occupancy, minor_day, 0, 0, accounting_memo
        )
        new += _partial_accounting_cost(
            daily_occupancy, major_day - 1, 0, major_diff, accounting_memo
        )
        old += _partial_accounting_cost(
            daily_occupancy, major_day - 1, 0, 0, accounting_memo
        )
    return old, new


@njit
def _delta_move_family_cost(
    prediction, daily_occupancy, f, dst_day, family_size, penalty_memo, accounting_memo
):
    src_day = prediction[f]
    if src_day == dst_day:
        return 0

    cp = family_size[f]

    old, new = _delta_swap_cost(
        daily_occupancy, src_day, dst_day, cp, 0, penalty_memo, accounting_memo
    )
    old += penalty_memo[f, src_day]
    new += penalty_memo[f, dst_day]
    return new - old


@njit
def _delta_swap_family_cost(
    prediction, daily_occupancy, f1, f2, family_size, penalty_memo, accounting_memo
):
    d1 = prediction[f1]
    d2 = prediction[f2]
    if d1 == d2:
        return 0

    c1 = family_size[f1]
    c2 = family_size[f2]

    old, new = _delta_swap_cost(
        daily_occupancy, d1, d2, c1, c2, penalty_memo, accounting_memo
    )
    old += penalty_memo[f1, d1] + penalty_memo[f2, d2]
    new += penalty_memo[f1, d2] + penalty_memo[f2, d1]
    return new - old


def build_cost_function(data, max_occupancy=300, min_occupancy=125):
    family_size = data.n_people.values
    days_array = np.arange(N_DAYS, 0, -1)

    # Precompute matrices needed for our cost function
    penalty_memo = create_penalty_memo(data)
    accounting_memo = create_accounting_memo()

    @njit
    def total_cost(prediction):
        return _total_cost(
            prediction,
            family_size=family_size,
            days_array=days_array,
            penalty_memo=penalty_memo,
            accounting_memo=accounting_memo,
            max_occupancy=max_occupancy,
            min_occupancy=min_occupancy,
        )

    @njit
    def delta_move_cost(prediction, daily_occupancy, f, dst_day):
        return _delta_move_family_cost(
            prediction,
            daily_occupancy,
            f,
            dst_day,
            family_size,
            penalty_memo,
            accounting_memo,
        )

    @njit
    def delta_swap_cost(prediction, daily_occupancy, f1, f2):
        return _delta_swap_family_cost(
            prediction,
            daily_occupancy,
            f1,
            f2,
            family_size,
            penalty_memo,
            accounting_memo,
        )

    return total_cost, delta_move_cost, delta_swap_cost
