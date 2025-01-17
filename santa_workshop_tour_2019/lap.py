from lap import lapjv
import numpy as np
from .cost import create_penalty_memo
from .util import group_by_day, group_by_family_size
import random
from collections import deque


def _concat_source_ids(id_groups, target_size):
    target_sizes = [*range(2, target_size - 1), target_size]
    concat_ids = sum((id_groups.get(size, []) for size in target_sizes), [])
    random.shuffle(concat_ids)
    return concat_ids


def _create_specified_family_size_groups(fam_ids, family_size, dst_size):
    fam_id_groups = group_by_family_size(fam_ids, family_size)
    concat_source_ids = _concat_source_ids(fam_id_groups, dst_size)

    qs = {i: deque([]) for i in range(2, dst_size + 1)}
    for cur_id in concat_source_ids:
        cur_size = family_size[cur_id]
        for prev_size in range(dst_size - cur_size, cur_size - 1, -cur_size):
            if 0 < len(qs[prev_size]):
                prev_ids = qs[prev_size].popleft()
                new_size = cur_size + prev_size
                new_ids = (cur_id, *prev_ids)
                qs[new_size].append(new_ids)
                break
        else:
            qs[cur_size].append((cur_id,))
    return list(qs[dst_size])


def _create_weights(prediction, fam_id_groups, penalty_memo):
    weights = np.zeros((len(fam_id_groups), len(fam_id_groups)), dtype=np.int64)
    for i in range(len(fam_id_groups)):
        for j in range(len(fam_id_groups)):
            if i == j:
                continue
            id_group_i = fam_id_groups[i]
            day_i = prediction[id_group_i[0]]

            id_group_j = fam_id_groups[j]
            day_j = prediction[id_group_j[0]]

            weights[i, j] = sum(
                [
                    penalty_memo[id_i, day_j] - penalty_memo[id_i, day_i]
                    for id_i in id_group_i
                ]
            )
    return weights


def build_family_size_lap(data):
    family_size = data.n_people.values
    penalty_memo = create_penalty_memo(data)

    def _family_size_lap(prediction, target_size):
        new = prediction.copy()

        families_per_day = group_by_day(prediction)
        id_groups = sum(
            [
                _create_specified_family_size_groups(fam_ids, family_size, target_size)
                for fam_ids in families_per_day.values()
            ],
            [],
        )

        weights = _create_weights(prediction, id_groups, penalty_memo)
        _, col, _ = lapjv(weights)

        for i in range(len(col)):
            new_day = prediction[id_groups[col[i]][0]]
            for j in id_groups[i]:
                new[j] = new_day

        return new

    return _family_size_lap
