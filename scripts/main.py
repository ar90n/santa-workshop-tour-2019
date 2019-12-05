from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function
from santa_workshop_tour_2019.const import cols, days
from santa_workshop_tour_2019.lap import solve as solve_lap
from santa_workshop_tour_2019.optim import greedy

data = io.load_data()

import numpy as np
occupancies = np.zeros(101, dtype=np.int)
occupancies[1:] = 50
best = solve_lap(data, occupancies)


i = 0
choice_dict = data[cols].to_dict()
total_cost = build_cost_function(data)
while i <= 15:
    best, score = greedy(best, choice_dict, total_cost)
    print(f"Score: {score}")
    i += 1

io.save_result(best, score)
