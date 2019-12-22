import numpy as np
from .const import cols
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
