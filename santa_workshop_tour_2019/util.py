import random
import numpy as np


def group_by_family_size(fam_ids, family_size):
    group = {}
    for fam_id in fam_ids:
        fam_size = family_size[fam_id]
        group.setdefault(fam_size, []).append(fam_id)
    return group

def group_by_day(prediction):
    fams = {}
    for fam_id in range(len(prediction)):
        day = prediction[fam_id]
        fams.setdefault(day, []).append(fam_id)
    return {k: np.array(v, dtype=np.int64) for k, v in fams.items()}


def non_adj_famply_sampling(prediction):
    families_per_day = group_by_day(prediction)

    days = set(range(1, 101))
    fam_ids = []
    #while 0 < len(days):
    while 95 < len(days):
        focus_day = random.choice(tuple(days))
        fam_id = random.choice(families_per_day[focus_day])
        fam_ids.append(fam_id)

        days.remove(focus_day)
        if focus_day - 1 in days:
            days.remove(focus_day - 1)
        if focus_day + 1 in days:
            days.remove(focus_day + 1)
    return fam_ids
