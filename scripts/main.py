from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function
from santa_workshop_tour_2019.const import cols, days
from santa_workshop_tour_2019.lap import solve as solve_lap
from santa_workshop_tour_2019.optim import build_greedy_move_func, build_greedy_swap_func, stochastic_product_search

data = io.load_data()

import numpy as np
occupancies = np.zeros(101, dtype=np.int)
occupancies[1:] = 50
best = solve_lap(data, occupancies)


i = 0
total_cost, delta_move_cost, delta_swap_cost = build_cost_function(data)

family_size = data.n_people.values
N = len(best)
daily_occupancy = np.zeros(101, dtype=np.int64)
for j in range(N):
    daily_occupancy[best[j]] += family_size[j]

greedy_move = build_greedy_move_func(data, delta_move_cost)
greedy_swap = build_greedy_swap_func(data, delta_swap_cost)

while i <= 15:
    best, daily_occupancy = greedy_move(best, daily_occupancy)
    score = total_cost(best)
    print(f"Score0: {score}")

    best, daily_occupancy = greedy_swap(best, daily_occupancy)
    score = total_cost(best)
    print(f"Score1: {score}")
    i += 1

io.save_result(best, score)
