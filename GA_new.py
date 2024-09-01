import random
from random import randrange
from time import time

# Replace these with distance matrix
cities = {0: 'City0', 1: 'City1', 2: 'City2', 3: 'City3', 4: 'City4', 5: 'City5', 6: 'City6', 7: 'City7'}
distances = {
    0: [float('inf'), 2, 3, 4, 5, 6, 7],
    1: [2, float('inf'), 8, 9, 10, 11, 12],
    2: [3, 8, float('inf'), 13, 14, 15, 16],
    3: [4, 9, 13, float('inf'), 17, 18, 19],
    4: [5, 10, 14, 17, float('inf'), 20, 21],
    5: [6, 11, 15, 18, 20, float('inf'), 22],
    6: [7, 12, 16, 19, 21, 22, float('inf')]
}

class Problem_Genetic(object):
    def __init__(self, genes, individuals_length, decode, fitness):
        self.genes = genes
        self.individuals_length = individuals_length
        self.decode = decode
        self.fitness = fitness

    def mutation(self, chromosome, prob):
        def inversion_mutation(chromosome_aux):
            chromosome = chromosome_aux
            index1 = randrange(0, len(chromosome))
            index2 = randrange(index1, len(chromosome))
            chromosome_mid = chromosome[index1:index2]
            chromosome_mid.reverse()
            chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
            return chromosome_result

        aux = []
        for _ in range(len(chromosome)):
            if random.random() < prob:
                aux = inversion_mutation(chromosome)
        return aux

    def crossover(self, parent1, parent2):
        def process_gen_repeated(copy_child1, copy_child2):
            count1 = 0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in parent1[pos:]:  # Choose next available gen
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2 += 1
                count1 += 1
            count1 = 0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:  # If need to fix repeated gen
                    count2 = 0
                    for gen2 in parent2[pos:]:  # Choose next available gen
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2 += 1
                count1 += 1
            return [child1, child2]

        pos = random.randrange(1, self.individuals_length - 1)
        child1 = parent1[:pos] + parent2[pos:]
        child2 = parent2[:pos] + parent1[pos:]

        return process_gen_repeated(child1, child2)

def decodeTSP(chromosome):
    lista = []
    for i in chromosome:
        lista.append(cities.get(i))
    return lista

def penalty(chromosome):
    actual = chromosome
    value_penalty = 0
    for i in actual:
        times = 0
        times = actual.count(i)
        if times > 1:
            value_penalty += 100 * abs(times - len(actual))
    return value_penalty

def fitnessTSP(chromosome):
    def distanceTrip(index, city):
        w = distances.get(index)
        return w[city]

    actualChromosome = list(chromosome)
    fitness_value = 0
    count = 0

    # Penalty for a city repetition inside the chromosome
    penalty_value = penalty(actualChromosome)

    for i in chromosome:
        if count == 7:
            nextCity = actualChromosome[0]
        else:
            temp = count + 1
            nextCity = actualChromosome[temp]

        fitness_value += distanceTrip(i, nextCity) + 50 * penalty_value
        count += 1

    return fitness_value

def genetic_algorithm_t(Problem_Genetic, k, opt, ngen, size, ratio_cross, prob_mutate):
    def initial_population(Problem_Genetic, size):
        def generate_chromosome():
            chromosome = []
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome

        return [generate_chromosome() for _ in range(size)]

    def new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate):
        def tournament_selection(Problem_Genetic, population, n, k, opt):
            winners = []
            for _ in range(int(n)):
                elements = random.sample(population, k)
                winners.append(opt(elements, key=Problem_Genetic.fitness))
            return winners

        def cross_parents(Problem_Genetic, parents):
            childs = []
            for i in range(0, len(parents), 2):
                childs.extend(Problem_Genetic.crossover(parents[i], parents[i + 1]))
            return childs

        def mutate(Problem_Genetic, population, prob):
            for i in population:
                Problem_Genetic.mutation(i, prob)
            return population

        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,
                                tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations

        return new_generation

    population = initial_population(Problem_Genetic, size)
    n_parents = round(size * ratio_cross)
    n_parents = (n_parents if n_parents % 2 == 0 else n_parents - 1)
    n_directs = int(size - n_parents)

    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)

    bestChromosome = opt(population, key=Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print("Solution:", (genotype, Problem_Genetic.fitness(bestChromosome)))
    return (genotype, Problem_Genetic.fitness(bestChromosome))

def genetic_algorithm_t2(Problem_Genetic, k, opt, ngen, size, ratio_cross, prob_mutate, dictionary):
    def initial_population(Problem_Genetic, size):

        def generate_chromosome():
            chromosome = []
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            # Adding to dictionary new generation
            dictionary[str(chromosome)] = 1
            return chromosome

        return [generate_chromosome() for _ in range(size)]

    def new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate):
        def tournament_selection(Problem_Genetic, population, n, k, opt):
            winners = []
            for _ in range(int(n)):
                elements = random.sample(population, k)
                winners.append(opt(elements, key=Problem_Genetic.fitness))
            for winner in winners:
                # For each winner, if exists in dictionary, we increase his age
                if str(winner) in dictionary:
                    dictionary[str(winner)] = dictionary[str(winner)] + 1
                # Else we need to initialize in dictionary
                else:
                    dictionary[str(winner)] = 1
            return winners

        def cross_parents(Problem_Genetic, parents):
            childs = []
            for i in range(0, len(parents), 2):
                childs.extend(Problem_Genetic.crossover(parents[i], parents[i + 1]))
                # Each time that some parent is crossed we add their two sons to dictionary
                if str(parents[i]) not in dictionary:
                    dictionary[str(parents[i])] = 1
                dictionary[str(childs[i])] = dictionary[str(parents[i])]
                # ...and remove their parents
                del dictionary[str(parents[i])]
            return childs

        def mutate(Problem_Genetic, population, prob):
            j = 0
            copy_population = population
            for crom in population:
                Problem_Genetic.mutation(crom, prob)
                # Each time that some parent is crossed
                if str(crom) in dictionary:
                    # We add the new chromosome mutated
                    dictionary[str(population[j])] = dictionary[str(crom)]
                    # Then we remove the parent, because his mutated has been added.
                    del dictionary[str(copy_population[j])]
                    j += j
            return population

        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic, tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        # Adding new generation of mutants to dictionary.
        for ind in new_generation:
            age = 0
            if str(ind) in dictionary:
                age += 1
                dictionary[str(ind)] += 1
            else:
                dictionary[str(ind)] = 1
        return new_generation

    population = initial_population(Problem_Genetic, size)
    n_parents = round(size * ratio_cross)
    n_parents = (n_parents if n_parents % 2 == 0 else n_parents - 1)
    n_directs = size - n_parents

    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)

    bestChromosome = opt(population, key=Problem_Genetic.fitness)
    print("Chromosome: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print("Solution:", (genotype, Problem_Genetic.fitness(bestChromosome)), dictionary[(str(bestChromosome))],
          " generations of winners parents.")
    return (genotype, Problem_Genetic.fitness(bestChromosome)
            + dictionary[(str(bestChromosome))] * 50)  # Updating fitness with age too

# Execute the TSP function
k = 5  # Replace with the desired number of runs
TSP(k)
