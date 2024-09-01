import numpy as np
import heapq
from copy import deepcopy
import time  # Import the time module

class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    def branchAndBound(self):
        results = {}
        count = 0
        max_queue_size = 0
        total_states_created = 0
        total_states_pruned = 0
        initial_rough_bound = 0.0
        self.BSSF = np.inf
        cities = self._scenario.getCities()
        ncities = len(cities)
        priority_queue = []
        heapq.heapify(priority_queue)
        row_reduced_matrix = deepcopy(cities)

        for i in range(ncities):
            for j in range(ncities):
                if i != j:
                    row_reduced_matrix[i][j] = cities[i][j]

        row_reduced_matrix, initial_rough_bound = self.reduce_matrix(row_reduced_matrix, ncities)

        if initial_rough_bound == 0.0:
            raise Exception("Initial Rough Bound cannot be 0 (unless everything is unreachable)")

        results['soln'] = None

        for city_index in range(ncities):
            initial_state = State(initial_rough_bound, [city_index], row_reduced_matrix, 0)
            heapq.heappush(priority_queue, initial_state)

        start_time = time.time()
        while len(priority_queue) > 0:
            current_state = heapq.heappop(priority_queue)
            count += 1
            if current_state.bound >= self.BSSF:
                total_states_pruned += 1
                continue

            if len(current_state.path) == ncities:
                last_city = current_state.path[-1]
                cost = self._scenario.getCities()[last_city][current_state.path[0]]
                current_state.cost += cost
                if current_state.cost < self.BSSF:
                    self.BSSF = current_state.cost
                    results['soln'] = TSPSolution(self._scenario.getCities(), current_state.path)
            else:
                last_city = current_state.path[-1]
                for next_city in range(ncities):
                    if next_city == last_city or next_city in current_state.path:
                        continue

                    new_matrix = deepcopy(current_state.row_reduced_matrix)
                    cost = self._scenario.getCities()[last_city][next_city]
                    new_matrix, bound = self.reduce_matrix(new_matrix, ncities)
                    new_bound = bound + current_state.bound + cost

                    new_path = deepcopy(current_state.path)
                    new_path.append(next_city)

                    new_state = State(new_bound, new_path, new_matrix, current_state.cost + cost)
                    heapq.heappush(priority_queue, new_state)

            if len(priority_queue) > max_queue_size:
                max_queue_size = len(priority_queue)

            total_states_created += 1

        end_time = time.time()
        results['time'] = end_time - start_time
        results['count'] = count
        results['max'] = max_queue_size
        results['total'] = total_states_created
        results['pruned'] = total_states_pruned
        results['cost'] = self.BSSF
        return results

    def reduce_matrix(self, matrix, ncities):
        bound = 0
        # row reduction
        for i in range(ncities):
            min_val = np.min(matrix[i][matrix[i] != np.inf])
            if np.isinf(min_val):
                continue
            bound += min_val
            matrix[i][matrix[i] != np.inf] -= min_val

        # column reduction
        for j in range(ncities):
            min_val = np.min(matrix[:, j][matrix[:, j] != np.inf])
            if np.isinf(min_val):
                continue
            bound += min_val
            matrix[:, j][matrix[:, j] != np.inf] -= min_val

        return matrix, bound


class TSPSolution:
    def __init__(self, cities, route):
        self.route = route
        self.cost = 0
        if cities is not None and route:
            for i in range(len(route) - 1):
                self.cost += cities[route[i]][route[i + 1]]
            self.cost += cities[route[-1]][route[0]]


class TSPScenario:
    def __init__(self):
        self.cities = []

    def setCities(self, cityList):
        self.cities = cityList

    def getCities(self):
        return self.cities


class State:
    def __init__(self, bound, path, row_reduced_matrix, cost):
        self.bound = bound
        self.path = path
        self.row_reduced_matrix = row_reduced_matrix
        self.cost = cost

    def __lt__(self, other):
        return self.bound < other.bound


# Main part to solve TSP for 7 cities
if __name__ == "__main__":
    distances = np.array([
        [np.inf, 2, 3, 4, 5, 6, 7],
        [2, np.inf, 8, 9, 10, 11, 12],
        [3, 8, np.inf, 13, 14, 15, 16],
        [4, 9, 13, np.inf, 17, 18, 19],
        [5, 10, 14, 17, np.inf, 20, 21],
        [6, 11, 15, 18, 20, np.inf, 22],
        [7, 12, 16, 19, 21, 22, np.inf]
    ])

    solver = TSPSolver(None)
    scenario = TSPScenario()
    scenario.setCities(distances)
    solver.setupWithScenario(scenario)

    results = solver.branchAndBound()

    print("Cost of the best solution:", results['cost'])
    print("Time spent to find the best solution:", results['time'])
    print("Total number of solutions found:", results['count'])
    print("Maximum size of the priority queue:", results['max'])
    print("Total states created:", results['total'])
    print("Total states pruned:", results['pruned'])
    print("Best solution:", results['soln'].route)
