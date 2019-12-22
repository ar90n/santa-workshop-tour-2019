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
