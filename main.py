from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function, get_weights
from santa_workshop_tour_2019.const import MAX_OCCUPANCY, MIN_OCCUPANCY, cols, days

data = io.load_data()
submission = io.load_sample_submission()

family_size_dict = data[["n_people"]].to_dict()["n_people"]
choice_dict = data[cols].to_dict()
total_cost = build_cost_function(data)

weights = get_weights(data)
from lap import lapjv

least_cost, col, row = lapjv(weights)

# Start with the sample submission values
best = col // 50 + 1
start_score = total_cost(best)


from tqdm import tqdm

i = 0
new = best.copy()
while i <= 20:
    # loop over each family
    for fam_id in tqdm(range(len(best))):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f"choice_{pick}"][fam_id]
            temp = new.copy()
            temp[fam_id] = day  # add in the new pick
            if total_cost(temp) < start_score:
                new = temp.copy()
                start_score = total_cost(new)

    submission["assigned_day"] = new
    score = total_cost(new)
    print(f"Score: {score}")
    i += 1

submission.to_csv(f"submission_{score}.csv")
