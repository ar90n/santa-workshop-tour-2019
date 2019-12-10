import numpy as np
from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function, create_penalty_memo
from santa_workshop_tour_2019.const import FLAT_SLOTS
from santa_workshop_tour_2019.lap import build_family_lap, build_family_size_lap
from santa_workshop_tour_2019.optim import build_greedy_move_func, build_greedy_swap_func, stochastic_product_search

data = io.load_data()


total_cost, delta_move_cost, delta_swap_cost = build_cost_function(data)
greedy_move = build_greedy_move_func(data, delta_move_cost)
greedy_swap = build_greedy_swap_func(data, delta_swap_cost)
family_lap = build_family_lap(data)
family_size_lap = build_family_size_lap(data)

i = 0
best, daily_occupancy = family_lap(data, FLAT_SLOTS)
while i <= 15:
    best, daily_occupancy = greedy_move(best, daily_occupancy)
    score = total_cost(best)
    print(f"Score0: {score}")

    best, daily_occupancy = greedy_swap(best, daily_occupancy)
    score = total_cost(best)
    print(f"Score1: {score}")

    best = family_size_lap(best)
    score = total_cost(best)
    print(f"Score2: {score}")

    i += 1

io.save_result(best, score)
