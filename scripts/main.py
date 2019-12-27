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
mip = build_mip(data, choices=6, accounting_thresh=1024)

i = 0
best = np.array([int(v) for v in io.load_submission(Path("../input/santa2019temp/submission_70314.6291748948.csv"))["assigned_day"].to_list()], dtype=np.int64)
score, daily_occupancy = total_cost(best)
best_score = score
print(f"Score0: {score}")

while i <= 160:
    best, daily_occupancy = greedy_move(best, daily_occupancy)
    score, daily_occupancy = total_cost(best)
    print(f"Score1: {score}")

    best, daily_occupancy = greedy_swap(best, daily_occupancy)
    score, daily_occupancy = total_cost(best)
    print(f"Score2: {score}")

#    ords = list(range(2, 9))
#    random.shuffle(ords)
#    for s in ords:
#        best = family_size_lap(best, s)
#        score, daily_occupancy = total_cost(best)
#        print(f"Score3.{s}: {score}")

    h = int(22 - min(i // 5, 12))
    n = min(i // 3, 12)
    best = mip(best, daily_occupancy, n, h)
    score, daily_occupancy = total_cost(best)
    print(f"Score4: {score}")

    i += 1

    if score < best_score:
        best_score = score
        io.save_result(best, best_score)
