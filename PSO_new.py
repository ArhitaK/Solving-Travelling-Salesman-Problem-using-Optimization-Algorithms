import numpy as np
import time

class Particle:
    def __init__(self, num_cities, distance_matrix):
        self.position = np.random.permutation(num_cities)
        self.velocity = np.random.rand(num_cities)
        self.best_position = np.copy(self.position)
        self.fitness = self.calculate_fitness(distance_matrix)
        self.best_fitness = self.fitness

    def calculate_fitness(self, distance_matrix):
        total_distance = 0
        num_cities = len(self.position)
        for i in range(num_cities - 1):
            total_distance += distance_matrix[self.position[i]][self.position[i + 1]]
        # Return to the starting city
        total_distance += distance_matrix[self.position[-1]][self.position[0]]
        return total_distance

def update_velocity(particle, global_best_position, inertia_weight, c1, c2):
    inertia_term = inertia_weight * particle.velocity
    cognitive_term = c1 * np.random.rand() * (particle.best_position - particle.position)
    social_term = c2 * np.random.rand() * (global_best_position - particle.position)
    new_velocity = inertia_term + cognitive_term + social_term
    return new_velocity

def generate_unique_permutation(num_cities):
    while True:
        permutation = np.random.permutation(num_cities)
        if len(set(permutation)) == num_cities:
            return permutation

def dynamic_tsp_pso(distance_matrix, num_particles, num_iterations, c1, c2, w_max, w_min):
    num_cities = distance_matrix.shape[0]
    particles = [Particle(num_cities, distance_matrix) for _ in range(num_particles)]

    global_best_particle = min(particles, key=lambda x: x.best_fitness)

    start_time = time.time()

    for iteration in range(num_iterations):
        # Update inertia weight dynamically
        w = w_max - (iteration * (w_max - w_min) / num_iterations)

        for particle in particles:
            current_distance = particle.calculate_fitness(distance_matrix)
            if current_distance < particle.fitness:
                particle.fitness = current_distance
                particle.best_position = np.copy(particle.position)
            if current_distance < global_best_particle.best_fitness:
                global_best_particle = particle

        for particle in particles:
            particle.velocity = update_velocity(particle, global_best_particle.best_position, w, c1, c2)
            # Update particle position using the velocity
            new_position = np.argsort(particle.position + particle.velocity)
            particle.position = generate_unique_permutation(num_cities)

    end_time = time.time()
    processing_time = end_time - start_time

    return global_best_particle.best_position, global_best_particle.best_fitness, processing_time

if __name__ == "__main__":
    # Example usage for a dynamic TSP with changing distances
    distances = np.array([
        [np.inf, 2, 3, 4, 5, 6, 7],
        [2, np.inf, 8, 9, 10, 11, 12],
        [3, 8, np.inf, 13, 14, 15, 16],
        [4, 9, 13, np.inf, 17, 18, 19],
        [5, 10, 14, 17, np.inf, 20, 21],
        [6, 11, 15, 18, 20, np.inf, 22],
        [7, 12, 16, 19, 21, 22, np.inf]
    ])

    num_particles = 20
    num_iterations = 100
    c1 = 2.0
    c2 = 2.0
    w_max = 0.9
    w_min = 0.4

    best_solution, best_distance, processing_time = dynamic_tsp_pso(distances, num_particles, num_iterations, c1, c2, w_max, w_min)

    print("Best TSP solution:", best_solution)
    print("Total distance:", best_distance)
    print("Processing Time:", processing_time, "seconds")
