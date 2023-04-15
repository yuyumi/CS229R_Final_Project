import math
import os
from datetime import datetime

import numpy as np
import random
from typing import Callable, List, Optional
from constants import *
from functions import *

random.seed(0)

inp0 = Gate(None, None, 0)
inp1 = Gate(None, None, 1)
inp2 = Gate(None, None, 2)
inp3 = Gate(None, None, 3)

# NAND(NAND(x, y), x) = not(not(x and y) and x) = (x and y) or not(x) = (not x or y)
# NAND(NAND(x, y), y) = not(not(x and y) and y) = (x and y) or not(y) = (x or not y)
# NAND(not x or y, x or not y) = XOR(x, y)
nand0 = Gate(inp0, inp1, 4)
s0 = Gate(inp0, nand0, 5)
s1 = Gate(inp1, nand0, 6)
xor0 = Gate(s0, s1, 7)

nand1 = Gate(inp2, inp3, 8)
s2 = Gate(inp2, nand1, 9)
s3 = Gate(inp3, nand1, 10)
xor1 = Gate(s2, s3, 11)

nxor0 = Gate(xor0, xor0, 12)
nxor1 = Gate(xor1, xor1, 13)

total_or = Gate(nxor0, nxor1, 14)

gates = [
    nand0,
    s0,
    s1,
    xor0,
    nand1,
    s2,
    s3,
    xor1,
    nxor0,
    nxor1,
    total_or
]

genes = ''.join([
    '0' + format(gate.input1.id, '04b') + format(gate.input2.id, '04b') for gate in gates
]) + format(14, '04b')

print(genes)
print(fitness(genes, goals[1]))


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
# print(fitness('1' * BITS_PER_GENOME, Function(
#     give_inp0,
#     give_inp0,
#     give_inp0
# )))

init_pop = [generate_random_genome() for _ in range(POPULATION_SIZE)]
init_goal = random.randint(0, 1)

pop = [*init_pop]
goal = init_goal

# Create log name using timestamp
time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = f'logs/log_{time_string}.txt'
# If folder doesn't exist create it
if not os.path.exists('logs'):
    os.makedirs('logs')
f = open(log_file, 'w+')

for g in range(GENERATIONS):
    # anotate the population with their fitness
    pop_fitness = [(genome, fitness(genome, goals[goal])) for genome in pop]

    avg_fitness = sum(fitness for genome, fitness in pop_fitness) / len(pop_fitness)

    if (g + 1) % LOG_STEP == 0:
        f.write(f'[Generation {g + 1}] Fitness: {avg_fitness} Goal: {goal}\n')
        print(f'[Generation {g + 1}] Fitness: {avg_fitness} Goal: {goal}')

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

    if CHANGE_GOAL and (g + 1) % GOAL_CHANGE_T == 0:
        goal = random.randint(0, 1)
