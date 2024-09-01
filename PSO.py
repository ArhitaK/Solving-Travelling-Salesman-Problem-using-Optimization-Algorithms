import numpy as np
import time

class Particle:
    def __init__(self, num_cities):
        self.position = np.random.permutation(num_cities)
        self.velocity = np.random.permutation(num_cities)
        self.best_position = self.position.copy()
        self.best_cost = np.inf

def calculate_cost(solution, distance_matrix):
    cost = 0
    for i in range(len(solution) - 1):
        cost += distance_matrix[solution[i]][solution[i+1]]
    cost += distance_matrix[solution[-1]][solution[0]]  # Return to the starting city
    return cost

def update_velocity(particle, global_best_position, inertia_weight, cognitive_weight, social_weight):
    inertia_term = inertia_weight * particle.velocity
    cognitive_term = cognitive_weight * np.random.rand() * (particle.best_position - particle.position)
    social_term = social_weight * np.random.rand() * (global_best_position - particle.position)
    return inertia_term + cognitive_term + social_term

def pso_tsp(distance_matrix, num_particles=30, max_iterations=100, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5):
    num_cities = len(distance_matrix)
    particles = [Particle(num_cities) for _ in range(num_particles)]

    global_best_position = None
    global_best_cost = np.inf

    start_time = time.time()

    for _ in range(max_iterations):
        for particle in particles:
            cost = calculate_cost(particle.position, distance_matrix)

            if cost < particle.best_cost:
                particle.best_cost = cost
                particle.best_position = particle.position.copy()

            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_position, inertia_weight, cognitive_weight, social_weight)
            particle.position = np.argsort(particle.position + particle.velocity)

    end_time = time.time()
    processing_time = end_time - start_time

    return global_best_position, global_best_cost, processing_time

# Example distance matrix
distances = np.array([
    [np.inf, 2, 3, 4, 5, 6, 7],
    [2, np.inf, 8, 9, 10, 11, 12],
    [3, 8, np.inf, 13, 14, 15, 16],
    [4, 9, 13, np.inf, 17, 18, 19],
    [5, 10, 14, 17, np.inf, 20, 21],
    [6, 11, 15, 18, 20, np.inf, 22],
    [7, 12, 16, 19, 21, 22, np.inf]
])

# Solve TSP using PSO
best_solution, best_cost, processing_time = pso_tsp(distances)

print("Best TSP Solution:", best_solution)
print("Cost of the Best Solution:", best_cost)
print("Processing Time:", processing_time, "seconds")
