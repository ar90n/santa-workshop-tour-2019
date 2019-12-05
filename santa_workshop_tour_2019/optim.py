import numpy as np
import tqdm
from itertools import product
import random

def greedy(best, choice_dict, cost_fun, random_state=2019):
    # loop over each family
    start_score = cost_fun(best)
    new = best.copy()

    np.random.seed(random_state)

    ll = random.sample(list(range(len(best))), len(best))
    for fam_id in ll:
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f"choice_{pick}"][fam_id]
            temp = new.copy()
            temp[fam_id] = day  # add in the new pick
            if cost_fun(temp) < start_score:
                new = temp.copy()
                start_score = cost_fun(new)
    return new, start_score


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
        choice_matrix[:, i] = [f for _, f in sorted(choice_dict[f'choice_{i}'].items())]

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
