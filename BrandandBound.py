import itertools
import time
import numpy as np

def tsp_branch_and_bound(graph):
    def bound(path):
        cost = 0
        last = path[-1]
        remaining = [city for city in range(len(graph)) if city not in path]

        for city in remaining:
            cost += graph[last][city]

        return cost

    def tsp_recursive(path, cost):
        nonlocal optimal_path, optimal_cost

        if len(path) == len(graph):
            cost += graph[path[-1]][path[0]]
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_path = path[:]
            return

        for city in range(len(graph)):
            if city not in path:
                new_path = path + [city]
                new_cost = cost + graph[path[-1]][city]

                if bound(new_path) < optimal_cost:
                    tsp_recursive(new_path, new_cost)

    start_time = time.time()
    optimal_path = []
    optimal_cost = float('inf')

    tsp_recursive([0], 0)

    end_time = time.time()
    processing_time = end_time - start_time

    return optimal_path, optimal_cost, processing_time

# Example graph representing distances between cities
distances = np.array([
    [np.inf, 2, 3, 4, 5, 6, 7],
    [2, np.inf, 8, 9, 10, 11, 12],
    [3, 8, np.inf, 13, 14, 15, 16],
    [4, 9, 13, np.inf, 17, 18, 19],
    [5, 10, 14, 17, np.inf, 20, 21],
    [6, 11, 15, 18, 20, np.inf, 22],
    [7, 12, 16, 19, 21, 22, np.inf]
])

optimal_path, optimal_cost, processing_time = tsp_branch_and_bound(distances)

print("Optimal Path:", optimal_path)
print("Minimal Cost:", optimal_cost)
print("Processing Time:", processing_time, "seconds")
