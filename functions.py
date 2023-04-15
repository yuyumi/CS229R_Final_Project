import math
import asyncio

import numpy as np
import random
from typing import Callable, List, Optional, Tuple
from constants import *

BinFunc = Callable[[int, int], int]

# MODAL CODE
import modal.aio
np_image = modal.Image.debian_slim().pip_install("numpy")
stub = modal.aio.AioStub("fitness-calc")

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
    return ''.join(random.choice('01') for _ in range(BITS_PER_GENOME))


def split_genome_into_genes(genome: str) -> list[list[str]]:
    # gates are `BITS_PER_GENE` bits each
    # output is 4 bits
    return [
        [genome[i], genome[i + 1:i + BITS_PER_GATE + 1], genome[i + BITS_PER_GATE + 1:i + BITS_PER_GENE]]
        for i in range(0, len(genome) - 4, BITS_PER_GENE)
    ] + [[genome[-4:]]]


def genes_to_output(genes: list[list[str]], input) -> float:
    outp_gate = int(genes[-1][0], 2)
    return eval(outp_gate, genes, input)


def fitness(genome: str, goal: Function) -> float:
    # Compute losses over all possible inputs
    # fitness = 16
    total_losses= 0
    for i in range(16):
        actual = goal.eval(format(i, '04b'))
        for _ in range(BALDWIN_ITERS):
            genes = split_genome_into_genes(genome)
            for j in range(len(genes)):
                if genes[j][0] == '1':
                    genes[j][1] = format(np.random.randint(0, 16), '04b')
                    genes[j][2] = format(np.random.randint(0, 16), '04b')
            est = genes_to_output(genes, format(i, '04b'))
            total_losses += int(est != actual)
        # fitness -= int(est != actual)
    # fitness /= 16
    return 1-(total_losses/(16*BALDWIN_ITERS))

@stub.function()
def calc_fitness(genomes: List[str], goal: Function) -> List[Tuple[str, float]]:
    return [(genome, fitness(genome, goal)) for genome in genomes]

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


# XOR is AND(NAND, OR)
class Gate:

    def __init__(self, input1: Optional["Gate"], input2: Optional["Gate"], id: int):
        self.input1 = input1
        self.input2 = input2
        self.id = id

@stub.function(image=np_image)
def select_elite(pop_fitness):
    """
    Using the Elite selection method, best individuals
    """
    pop_fitness = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
    pop, fitnesses = zip(*pop_fitness)
    fitnesses = np.array(fitnesses)

    cum_exp_fitnesses = np.cumsum(EXP_FACTOR * exp_fitnesses)

    selected = []
    for _ in range(GRADUATION_SIZE):
        r = random.uniform(0, cum_exp_fitnesses[-1])
        idx = np.searchsorted(cum_exp_fitnesses, r)
        selected.append(pop[idx])

    return selected


def crossover(parent1, parent2):
    # With probability CROSSOVER_P, crossover two genes from parent1 and parent2
    if random.random() > CROSSOVER_P:
        return parent1, parent2
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


def mutate(genome: str):
    return ''.join([
        random.choice('01') if random.random() < MUTATION_P else genome[i]
        for i in range(len(genome))
    ])