import numpy as np
import time

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iteration, decay, alpha=1, beta=2):
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iteration = n_iteration
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iteration):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.n_best, self.distances)

            self.pheromone * self.decay

            self.intensify_pheromone(all_paths, self.distances)

            self.pheromone * self.decay

            self.pheromone[self.pheromone < 0.01] = 0.01

            current_shortest_path = min(all_paths, key=lambda x: x[1])

            if current_shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = current_shortest_path

        return all_time_shortest_path

    def spread_pheromone(self, all_paths, n_best, distances):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / distances[move]

    def intensify_pheromone(self, all_paths, distances):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:1]:
            for move in path:
                self.pheromone[move] += 1.0 / distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(len(path)-1):
            total_dist += self.distances[path[i]][path[i+1]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = [0]  # Starting from city 0
            for j in range(len(self.distances)-1):
                next_city = self.pick_next_city(path)
                path.append(next_city)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def pick_next_city(self, path):
        pheromone = self.pheromone[path[-1]]
        visibility = 1 / (np.array(self.distances[path[-1]]) + 1e-10)
        probabilities = (pheromone ** self.alpha) * (visibility ** self.beta)
        probabilities[path] = 0  # Exclude already visited cities
        probabilities /= probabilities.sum()
        return np.random.choice(self.all_inds, p=probabilities)

# Example usage:

distances = np.array([
    [np.NINF, 2, 3, 4, 5, 6, 7],
    [2, np.NINF, 8, 9, 10, 11, 12],
    [3, 8, np.NINF, 13, 14, 15, 16],
    [4, 9, 13, np.NINF, 17, 18, 19],
    [5, 10, 14, 17, np.NINF, 20, 21],
    [6, 11, 15, 18, 20, np.NINF, 22],
    [7, 12, 16, 19, 21, 22, np.NINF]
])

ant_colony = AntColony(distances, 100, 1, 100, 0.99, alpha=1, beta=5)

start_time = time.time()
shortest_path = ant_colony.run()
end_time = time.time()

processing_time = end_time - start_time

print("Shortest Path:", shortest_path[0])
print("Shortest Distance:", shortest_path[1])
print("Processing Time:", processing_time, "seconds")
