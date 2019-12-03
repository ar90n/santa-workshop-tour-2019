def greedy(best, choice_dict, cost_fun):
    # loop over each family
    start_score = cost_fun(best)
    new = best.copy()
    for fam_id in range(len(best)):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f"choice_{pick}"][fam_id]
            temp = new.copy()
            temp[fam_id] = day  # add in the new pick
            if cost_fun(temp) < start_score:
                new = temp.copy()
                start_score = cost_fun(new)
    return new, start_score
