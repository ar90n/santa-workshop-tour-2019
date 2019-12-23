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

data = io.load_data()

total_cost, delta_move_cost, delta_swap_cost = build_cost_function(data)
greedy_move = build_greedy_move_func(data, delta_move_cost)
greedy_swap = build_greedy_swap_func(data, delta_swap_cost)
family_size_lap = build_family_size_lap(data)
init = build_init_solver(data)
mip = build_mip(data, choices=6, accounting_thresh=1024)

i = 0
best = init(choices=6, accounting_thresh=4096)
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

    ords = list(range(2, 9))
    random.shuffle(ords)
    for s in ords:
        best = family_size_lap(best, s)
        score, daily_occupancy = total_cost(best)
        print(f"Score3.{s}: {score}")

    best = mip(best, daily_occupancy)
    score, daily_occupancy = total_cost(best)
    print(f"Score4: {score}")

    i += 1

    if score < best_score:
        best_score = score
        io.save_result(best, best_score)
