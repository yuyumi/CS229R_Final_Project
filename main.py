import asyncio
import math
import os
from datetime import datetime

import random
from typing import Callable, List, Optional, Tuple

USING_MODAL = False

# MODAL CODE
import modal.aio
np_image = modal.Image.debian_slim().pip_install("numpy")
stub = modal.aio.AioStub("fitness-calc")

random.seed(0)

gates = 11
bits_per_gate = 4
bits_per_gene = 1 + 2 * bits_per_gate  # 1 bit for if the gene can be Baldwin Effect'd
bits_per_genome = bits_per_gene * gates + 4

crossover_p = 0.5
mutation_p = 0.5 / bits_per_genome

population_size = 5000
graduation_size = 1500  # number of organisms that move to the next generation

generations = 100000
log_step = 10
goal_change_t = 20

exp_factor = 30

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

goals = [
    Function(bin_funcs[0], bin_funcs[0], bin_funcs[1]),
    Function(bin_funcs[0], bin_funcs[0], bin_funcs[2])
]


def generate_random_genome() -> str:
    return ''.join(random.choice('01') for _ in range(bits_per_genome))


def split_genome_into_genes(genome: str) -> list[list[str]]:
    # gates are `bits_per_gene` bits each
    # output is 4 bits
    return [
        [genome[i], genome[i + 1:i + bits_per_gate + 1], genome[i + bits_per_gate + 1:i + bits_per_gene]]
        for i in range(0, len(genome) - 4, bits_per_gene)
    ] + [[genome[-4:]]]


def genome_to_ouput(genome: str, input) -> float:
    genes = split_genome_into_genes(genome)
    outp_gate = int(genes[-1][0], 2)
    return eval(outp_gate, genes, input)


def eval(outp_gate, genes, input):
    val: List[Optional[int]] = [None] * 16
    vis = [False] * 16

    # Returns True if successful
    def dfs(node: int) -> bool:
        if vis[node]:
            return val[node] is not None
        vis[node] = True
        if node == 15:
            return False
        if node < 4:
            val[node] = int(input[node])
            return True

        input1 = int(genes[node - 4][1], 2)
        input2 = int(genes[node - 4][2], 2)
        if not dfs(input1) or not dfs(input2):
            return False
        val[node] = not (val[input1] and val[input2])
        return True

    if not dfs(outp_gate):  # If the DFS fails
        return -1

    return val[outp_gate]


def fitness(genome: str, goal: Function) -> float:
    # Compute losses over all possible inputs
    fitness = 16
    for i in range(16):
        est = genome_to_ouput(genome, format(i, '04b'))
        actual = goal.eval(format(i, '04b'))
        fitness -= int(est != actual)
    fitness /= 16
    return fitness


@stub.function()
def calc_fitness(genomes: List[str], goal: Function) -> List[Tuple[str, float]]:
    return [(genome, fitness(genome, goal)) for genome in genomes]


# XOR is AND(NAND, OR)
class Gate:

    def __init__(self, input1: Optional["Gate"], input2: Optional["Gate"], id: int):
        self.input1 = input1
        self.input2 = input2
        self.id = id


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
# print(fitness('1' * bits_per_genome, Function(
#     give_inp0,
#     give_inp0,
#     give_inp0
# )))


@stub.function(image=np_image)
def select_elite(pop_fitness):
    """
    Using the Elite selection method, best individuals
    """
    import numpy as np
    pop_fitness = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
    pop, fitnesses = zip(*pop_fitness)
    fitnesses = np.array(fitnesses)

    exp_fitnesses = math.e ** (exp_factor * fitnesses)
    cum_exp_fitnesses = np.cumsum(exp_fitnesses)

    selected = []
    for _ in range(graduation_size):
        r = random.uniform(0, cum_exp_fitnesses[-1])
        idx = np.searchsorted(cum_exp_fitnesses, r)
        selected.append(pop[idx])

    return selected


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

@stub.local_entrypoint()
async def main():
    init_pop = [generate_random_genome() for _ in range(population_size)]
    init_goal = random.randint(0, 1)

    pop = [*init_pop]
    goal = init_goal

    # Create log name using timestamp
    time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'logs/modal_log_{time_string}.txt'
    # If folder doesn't exist create it
    if not os.path.exists('logs'):
        os.makedirs('logs')
    f = open(log_file, 'w+')

    for g in range(generations):
        # anotate the population with their fitness
        if USING_MODAL:
            CHUNK_SIZE = 2500
            pop_fitness = sum(
                await asyncio.gather(
                    *[calc_fitness.call(pop[i:i + CHUNK_SIZE], goals[goal]) for i in range(0, len(pop), CHUNK_SIZE)]),
                []
            )
        else:
            pop_fitness = calc_fitness(pop, goals[goal])

        avg_fitness = sum(fitness for genome, fitness in pop_fitness) / len(pop_fitness)

        if (g + 1) % log_step == 0:
            f.write(f'[Generation {g + 1}] Fitness: {avg_fitness} Goal: {goal}\n')
            print(f'[Generation {g + 1}] Fitness: {avg_fitness} Goal: {goal}')

        selected_pop = await select_elite.call(pop_fitness) if USING_MODAL else select_elite(pop_fitness)

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
            goal = random.randint(0, 1)


if __name__ == '__main__':
    asyncio.run(main())
