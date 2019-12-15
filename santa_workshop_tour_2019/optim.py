import numpy as np
from itertools import product
import random
from santa_workshop_tour_2019.const import cols
from functools import partial
from numba import njit


def build_greedy_move_func(data, delta_move_cost_func):
    family_size = data.n_people.values
    choice_mat = data[cols].values

    @njit(fastmath=True)
    def _greedy_move_impl(best, daily_occupancy):
        # loop over each family
        new = best.copy()
        daily_occupancy = daily_occupancy.copy()

        fam_ids = np.array([i for i in range(len(best))])
        np.random.shuffle(fam_ids)
        for fam_id in fam_ids:
            for pick in range(10):
                day = choice_mat[fam_id, pick]
                if delta_move_cost_func(new, daily_occupancy, fam_id, day) < 0:
                    daily_occupancy[new[fam_id]] -= family_size[fam_id]
                    daily_occupancy[day] += family_size[fam_id]
                    new[fam_id] = day
        return new, daily_occupancy

    return _greedy_move_impl


def build_greedy_swap_func(data, delta_swap_cost_func):
    family_size = data.n_people.values

    @njit(fastmath=True)
    def _greedy_swap_impl(best, daily_occupancy):
        new = best.copy()
        daily_occupancy = daily_occupancy.copy()

        src_ids = np.array([i for i in range(len(best))])
        np.random.shuffle(src_ids)
        dst_ids = np.array([i for i in range(len(best))])
        np.random.shuffle(dst_ids)
        for src_fam_id in src_ids:
            for dst_fam_id in dst_ids:
                if src_fam_id == dst_fam_id:
                    continue

                if (
                    delta_swap_cost_func(new, daily_occupancy, src_fam_id, dst_fam_id)
                    < 0
                ):
                    daily_occupancy[new[src_fam_id]] += (
                        family_size[dst_fam_id] - family_size[src_fam_id]
                    )
                    daily_occupancy[new[dst_fam_id]] += (
                        family_size[src_fam_id] - family_size[dst_fam_id]
                    )
                    new[src_fam_id], new[dst_fam_id] = new[dst_fam_id], new[src_fam_id]
        return new, daily_occupancy

    return _greedy_swap_impl


def stochastic_product_search(
    best,
    choice_dict,
    cost_function,
    top_k,
    fam_size,
    disable_tqdm=False,
    verbose=10000,
    n_iter=500,
    random_state=2019,
):
    """
    original (np.array): The original day assignments.

    At every iterations, randomly sample fam_size families. Then, given their top_k
    choices, compute the Cartesian product of the families' choices, and compute the
    score for each of those top_k^fam_size products.
    """

    best = best.copy()
    best_score = cost_function(best)

    np.random.seed(random_state)

    choice_matrix = np.zeros((len(best), top_k), dtype=np.int64)
    for i in range(top_k):
        choice_matrix[:, i] = [f for _, f in sorted(choice_dict[f"choice_{i}"].items())]

    for i in range(n_iter):
        fam_indices = np.random.choice(range(len(best)), size=fam_size)
        changes = np.array(list(product(*choice_matrix[fam_indices, :].tolist())))

        for change in changes:
            new = best.copy()
            new[fam_indices] = change

            new_score = cost_function(new)

            if new_score < best_score:
                best_score = new_score
                best = new

        if new_score < best_score:
            best_score = new_score
            best = new

        if verbose and i % verbose == 0:
            print(f"Iteration #{i}: Best score is {best_score:.2f}")

    print(f"Final best score is {best_score:.2f}")
    return best, best_score
