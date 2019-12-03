from santa_workshop_tour_2019 import io
from santa_workshop_tour_2019.cost import build_cost_function
from santa_workshop_tour_2019.const import cols, days
from santa_workshop_tour_2019.lap import solve as solve_lap
from santa_workshop_tour_2019.optim import greedy

data = io.load_data()

best = solve_lap(data)

i = 0
choice_dict = data[cols].to_dict()
total_cost = build_cost_function(data)
while i <= 15:
    best, score = greedy(best, choice_dict, total_cost)
    print(f"Score: {score}")
    i += 1

io.save_result(best, score)
