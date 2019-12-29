from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function
from santa_workshop_tour_2019.lap import build_family_size_lap
from santa_workshop_tour_2019.greedy import (
    build_greedy_move_func,
    build_greedy_swap_func,
)
from santa_workshop_tour_2019.mip import build_init_solver, build_mip
from pathlib import Path
import random
import numpy as np

data = io.load_data()

total_cost, delta_move_cost, delta_swap_cost = build_cost_function(data)
greedy_move = build_greedy_move_func(data, delta_move_cost)
greedy_swap = build_greedy_swap_func(data, delta_swap_cost)
family_size_lap = build_family_size_lap(data)
mip = build_mip(data, choices=6, accounting_thresh=4096)

i = 0
best = np.array(
    [
        int(v)
        for v in io.load_submission(
            Path("../input/santa2019work/submission.csv")
        )["assigned_day"].to_list()
    ],
    dtype=np.int64,
)
score, daily_occupancy = total_cost(best)
best_score = score
print(f"Score0: {score}")

io.save_result(best)
while i <= 2:
    cur_best = mip(best, daily_occupancy, 30, 15)
    cur_score, cur_daily_occupancy = total_cost(cur_best)
    print(f"Score4: {cur_score}")

    i += 1

    if cur_score < best_score:
        best = cur_best
        daily_occupancy = cur_daily_occupancy
        best_score = cur_score
        io.save_result(cur_best)
