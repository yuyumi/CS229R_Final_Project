import os
from datetime import datetime

import numpy as np
import random
from typing import Callable

random.seed(0)

gates = 11
bits_per_gate = 4
bits_per_gene = 1 + 2 * bits_per_gate  # 1 bit for if the gene can be Baldwin Effect'd
bits_per_genome = bits_per_gene * gates + 4

crossover_p = 0.5
mutation_p = 0.7 / bits_per_genome

population_size = 1000
graduation_size = 300  # number of organisms that move to the next generation

generations = 10000
log_step = 1
goal_change_t = 20

change_goal = True

BinFunc = Callable[[int, int], int]


class Function:

    def __init__(self, f: BinFunc, g: BinFunc, h: BinFunc):
        self.f = f
        self.g = g
        self.h = h

    def eval(self, inputs: str) -> int:
        # split input into four bits
        bits = [int(inputs[i]) for i in range(0, len(inputs))]
        x = self.f(bits[0], bits[1])
        y = self.g(bits[2], bits[3])
        return self.h(x, y)


bin_funcs = [
    # XOR
    lambda x, y: x ^ y,
    # AND
    lambda x, y: x & y,
    # OR
    lambda x, y: x | y,
]


def generate_random_goal() -> Function:
    # Choose f, g, and h randomly from bin_Funcs
    return Function(random.choice(bin_funcs), random.choice(bin_funcs), random.choice(bin_funcs))


def generate_random_genome() -> str:
    return ''.join(random.choice('01') for _ in range(bits_per_genome))


def split_genome_into_genes(genome: str) -> list[str]:
    # gates are `bits_per_gene` bits each
    # output is 4 bits
    return [[genome[i], genome[i + 1:i + bits_per_gate + 1], genome[i + bits_per_gate + 1:i + bits_per_gene]] for i in
            range(0, len(genome) - 4, bits_per_gene)] + [[genome[-4:]]]


def fitness(genome: str, goal: Function) -> float:
    # Compute losses over all possible inputs
    fitness = 16
    for i in range(16):
        est = genome_to_ouput(genome, format(i, '04b'))
        actual = goal.eval(format(i, '04b'))
        fitness -= int(est != actual)
    fitness /= 16
    return fitness


def eval(outp_gate, genes, input):
    val = [None] * 16
    vis = [False] * 16

    # Returns True if successful
    def dfs(node) -> bool:
        if vis[node]:
            return val[node] is not None
        vis[node] = True
        if node == 15: return False
        if node < 4:
            val[node] = int(input[node])
            return True

        input1 = int(genes[node - 4][1], 2)
        input2 = int(genes[node - 4][2], 2)
        if not dfs(input1) or not dfs(input2):
            return False
        val[node] = not (val[input1] and val[input2])
        return False

    if not dfs(outp_gate):  # If the DFS fails
        return -1

    return val[outp_gate]


def genome_to_ouput(genome: str, input) -> float:
    genes = split_genome_into_genes(genome)
    return eval(int(genes[-1][0], 2), genes, input)


# print(genes_test)
# print(genome_to_ouput(test, '1010'))
# print(genome_to_ouput(test, '0010'))
# print(genome_to_ouput(test, '1110'))

# give_inp0 = lambda x, y: x
# print(fitness(test, Function(
#     give_inp0,
#     give_inp0,
#     give_inp0
# )))
# print(fitness('1' * bits_per_genome, Function(
#     give_inp0,
#     give_inp0,
#     give_inp0
# )))


def select_elite(pop_fitness, exp_weight=1):
    """
    Using the Elite selection method, best individuals
    """
    pop_fitness = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
    pop, fitnesses = zip(*pop_fitness)
    fitnesses = np.array(fitnesses)

    exp_fitnesses = fitnesses ** exp_weight
    cum_exp_fitnesses = np.cumsum(exp_fitnesses)

    selected = []
    for _ in range(graduation_size):
        r = random.uniform(0, cum_exp_fitnesses[-1])
        idx = np.searchsorted(cum_exp_fitnesses, r)
        selected.append(pop[idx])

    return selected


init_pop = [generate_random_genome() for _ in range(population_size)]
init_goal = generate_random_goal()


def crossover(parent1, parent2):
    # With probability crossover_p, crossover two genes from parent1 and parent2
    if random.random() > crossover_p:
        return parent1, parent2
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def mutate(genome: str):
    return ''.join([
        random.choice('01') if random.random() < mutation_p else genome[i]
        for i in range(len(genome))
    ])


pop = [*init_pop]
goal = init_goal

# Create log name using timestamp
time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = f'logs/log_{time_string}.txt'
# If folder doesn't exist create it
if not os.path.exists('logs'):
    os.makedirs('logs')
f = open(log_file, 'w+')

for g in range(generations):
    # anotate the population with their fitness
    pop_fitness = [(genome, fitness(genome, goal)) for genome in pop]

    avg_fitness = sum(fitness(genome, goal) for genome, _ in pop_fitness) / len(pop_fitness)

    if (g + 1) % log_step == 0:
        f.write(f'[Generation {g + 1}] Fitness: {avg_fitness}\n')

    selected_pop = select_elite(pop_fitness, exp_weight=4)

    # Crossover
    offspring = []
    for _ in range(len(pop) // 2):
        parent1 = random.choice(selected_pop)
        parent2 = random.choice(selected_pop)
        while parent2 == parent1:
            parent2 = random.choice(selected_pop)

        offspring1, offspring2 = crossover(parent1, parent2)
        offspring.append(offspring1)
        offspring.append(offspring2)

    # Mutation
    offspring = [mutate(genome) for genome in offspring]

    # Update the population for the next generation
    pop = offspring

    if change_goal and (g + 1) % goal_change_t == 0:
        goal = generate_random_goal()
