from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function
from santa_workshop_tour_2019.lap import (
    build_family_size_lap,
    build_non_adj_family_lap,
)
from santa_workshop_tour_2019.optim import (
    build_greedy_move_func,
    build_greedy_swap_func,
)
from santa_workshop_tour_2019.lp_mip import build_lp_mip, build_mip
from pathlib import Path
import random

data = io.load_data()

total_cost, delta_move_cost, delta_swap_cost = build_cost_function(data)
greedy_move = build_greedy_move_func(data, delta_move_cost)
greedy_swap = build_greedy_swap_func(data, delta_swap_cost)
family_size_lap = build_family_size_lap(data)
non_adj_family_lap = build_non_adj_family_lap(data)
lp_mip = build_lp_mip(data)
mip = build_mip(data)

i = 0
#best = lp_mip()
best = io.load_submission(Path('./submission_76934.51860155357.csv'))['assigned_day'].values
score, daily_occupancy = total_cost(best)
best_score = score
print(f"Score0: {score}")
#io.save_result(best, best_score)

best = mip(best, daily_occupancy)
score, daily_occupancy = total_cost(best)
print(f"Score1: {score}")
