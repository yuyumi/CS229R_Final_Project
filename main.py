import json
import math
import os
from datetime import datetime
import asyncio
import argparse

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

# print(genes)
# print(fitness(genes, goals[1]))


def save_checkpoint(pop, goal, gen, run_name):
    file_name = f'ckpts/{time_string}_gen_{gen}.json'
    with open(file_name, 'w+') as f:
        json.dump({
            'pop': pop,
            'goal': goal,
            'gen': gen
        }, f)

async def main():
    global CHANGE_GOAL
    global BALDWIN_ITERS
    
    # If folder doesn't exist create it
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--name', type=str, help='Name of run')
    parser.add_argument('--baldwin', type=str, help='Number of baldwin iters')
    parser.add_argument('--change_goal', type=bool, help='Whether to used varying goals')
    args = parser.parse_args()

    if args.baldwin:
        BALDWIN_ITERS = int(args.baldwin)
        print(f'Using {BALDWIN_ITERS} baldwin iterations')
    
    if args.change_goal:
        CHANGE_GOAL = True
    print("Change Goal:", CHANGE_GOAL)

    if args.checkpoint:
        with open(args.checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
        pop = checkpoint_data['pop']
        goal = checkpoint_data['goal']
        start_gen = checkpoint_data['gen']

        fname = args.checkpoint
    else:
        init_pop = [generate_random_genome() for _ in range(POPULATION_SIZE)]
        init_goal = random.randint(0, 1) if CHANGE_GOAL else 0
        pop = [*init_pop]
        goal = init_goal
        start_gen = 0

    # Create log name using timestamp
    run_name = args.name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'logs/log_{run_name}.txt'
    # assert that the file doesn't already exist
    # Assert that the log file doesn't already exist
    assert not os.path.exists(log_file), f"Log file {log_file} already exists"

    f = open(log_file, 'w+')

    times = []

    for g in range(start_gen, GENERATIONS):
        start_t = datetime.now()
        # anotate the population with their fitness
        pop_fitness = calc_fitness(pop, goals[goal])

        selected_pop = select_elite(pop_fitness)

        # Crossover
        offspring = []
        for _ in range(len(pop) // 2):
            parent1 = random.choice(selected_pop)
            parent2 = random.choice(selected_pop)
            while parent2 == parent1:
                parent2 = random.choice(selected_pop)
            offspring.extend(crossover(parent1, parent2))

        # Mutation
        offspring = [mutate(genome) for genome in offspring]

        end_t = datetime.now()
        times.append((end_t - start_t).total_seconds())

        if (g + 1) % LOG_STEP == 0:
            avg_fitness = sum(fitness for genome, fitness in pop_fitness) / len(pop_fitness)
            non_baldwin_fitness = calc_fitness(pop, goals[goal], 1)
            avg_baldwin_fitness = sum(fitness for genome, fitness in non_baldwin_fitness) / len(non_baldwin_fitness)

            pm = percent_maleable(pop)

            avg_time = sum(times[-LOG_STEP:]) / LOG_STEP

            report = f'[Generation {g + 1}] Fitness: {avg_fitness} No-Baldwin Fitness: {avg_baldwin_fitness} Maleable: {pm * 100}% Goal: {goal} Times: {avg_time}s'
            f.write(report + "\n")
            print(report)

        if CHANGE_GOAL and (g + 1) % GOAL_CHANGE_T == 0:
            goal = random.randint(0, 1)

        if (g + 1) % CHECKPOINT_T == 0:
            save_checkpoint(pop, goal, g + 1, run_name)

        pop = offspring


if __name__ == '__main__':
    asyncio.run(main())
