import numpy as np
import random
import time

class GeneticAlgorithmTSP:
    def __init__(self, distance_matrix, population_size=50, crossover_rate=0.8, mutation_rate=0.02, max_generations=100):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.permutation(self.num_cities) for _ in range(self.population_size)]

    def calculate_fitness(self, individual):
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[individual[i]][individual[i + 1]]
        total_distance += self.distance_matrix[individual[-1]][individual[0]]  # Return to the starting city
        return 1 / total_distance

    def select_parents(self):
        parents = random.sample(self.population, 2)
        parents.sort(key=lambda x: self.calculate_fitness(x), reverse=True)
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            start = random.randint(0, self.num_cities - 1)
            end = random.randint(start + 1, self.num_cities)
            child = [-1] * self.num_cities
            child[start:end] = parent1[start:end]
            remaining_cities = [city for city in parent2 if city not in child]
            index = 0
            for i in range(self.num_cities):
                if child[i] == -1:
                    child[i] = remaining_cities[index]
                    index += 1
            return child
        else:
            return parent1

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            indices = random.sample(range(self.num_cities), 2)
            individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

    def evolve(self):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def run(self):
        start_time = time.time()
        for generation in range(self.max_generations):
            self.evolve()
        best_individual = max(self.population, key=lambda x: self.calculate_fitness(x))
        best_fitness = self.calculate_fitness(best_individual)
        end_time = time.time()
        processing_time = end_time - start_time
        return best_individual, 1 / best_fitness, processing_time

# Example distance matrix
distances = np.array([
        [np.inf, 454, 317, 165, 528, 222, 223],
        [454, np.inf, 253, 291, 210, 325, 121],
        [317, 253, np.inf, 202, 226, 108, 158],
        [165, 219, 202, np.inf, 344, 94, 114],
        [528, 210, 222, 223, np.inf, 182, 247],
        [222, 325, 108, 94, 114, np.inf, 206],
        [223, 121, 158, 114, 247, 206, np.inf]
    ])

# Solve TSP using Genetic Algorithm
ga_solver = GeneticAlgorithmTSP(distances)
best_solution, best_cost, processing_time = ga_solver.run()

print("Best TSP Solution:", best_solution)
print("Cost of the Best Solution:", best_cost)
print("Processing Time:", processing_time, "seconds")
