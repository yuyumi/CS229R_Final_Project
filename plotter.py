import re
import matplotlib.pyplot as plt
import numpy as np

import sys

if len(sys.argv) < 3:
    print("Usage: python plotter.py <beta> <change_goal>")
    exit(1)

BETA = int(sys.argv[1])
CHANGE_GOAL = bool(int(sys.argv[2]))

prefix = f'baldwin_{BETA}_' if BETA > 1 else ''
goal = f'fvg' if CHANGE_GOAL else 'fixed'

# Format:
format = re.compile(r'\[Generation (\d+)\] Fitness: (\d+\.\d+) No-Baldwin Fitness: (\d+\.\d+) Maleable: (\d+\.\d+)% Goal: (\d) Times: (\d+\.\d+)s')

header = ['Generation', 'Fitness', 'Pure Fitness', '% Maleable', 'Goal', 'Time']

data_map: dict[int, list] = {}
data0_map = {}
data1_map = {}

# LOAD THE DATA
for i in range(1, 11):
    FILE = f'aws/logs/log_{prefix}{goal}_{i}.txt'
    with open(FILE, 'r', encoding='utf-8') as f:
        data = f.read()

    lines = data.split('\n')
    data = []
    for line in lines:
        match = format.match(line)
        if match:
            data.append(tuple(map(float, match.groups())))

    data = [d for d in data if d[0] <= 30000]

    # Only where Goal is 0
    goal0_data = np.array([d for d in data if d[4] == 0 and d[0] % 100 == 0], dtype=float)
    # goal0_data = np.array([d for d in data if d[2] == 0 and d[0] % 100 == 0], dtype=float)
    data0_map[i] = goal0_data

    if CHANGE_GOAL:
        goal1_data = np.array([d for d in data if d[4] == 1 and d[0] % 100 == 0], dtype=float)
        # goal1_data = np.array([d for d in data if d[2] == 1 and d[0] % 100 == 0], dtype=float)
        data1_map[i] = goal1_data

    data_map[i] = data


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


TITLE = (f"Fixed Goal" if not CHANGE_GOAL else f"Varying Goal") + (f" Baldwin (beta={BETA}) " if BETA > 1 else " ") + "Fitness over Generations"

def drawall():
    for i in range(1, 11):
        goal0_data = data0_map[i]
        if CHANGE_GOAL:
            goal1_data = data1_map[i]
        data = np.array(data_map[i])

        if BETA > 1:
            plt.plot(goal0_data[:, 0], goal0_data[:, 1], label='Baldwin G1', color='#ff000080')
        plt.plot(goal0_data[:, 0], goal0_data[:, 2], label='G1', color=('#ff000040' if BETA > 1 else "#ff000080"))

        if CHANGE_GOAL:
            if BETA > 1:
                plt.plot(goal1_data[:, 0], goal1_data[:, 1], label='Baldwin G2', color='#0000ff80')
            plt.plot(goal1_data[:, 0], goal1_data[:, 2], label='G2', color='#0000ff40'if BETA > 1 else "#0000ff80")

        if BETA > 1:
            window_size = 10  # Adjust this value for the desired moving average window size
            maleable_moving_avg = moving_average(data[:, 3] / 100, window_size)
            plt.plot(data[window_size - 1:, 0], maleable_moving_avg, label='Plasticity (Moving Avg)', color='#00aa00')

        plt.title(TITLE)

        plt.xlabel('Generations')
        plt.ylabel('Fitness')

        plt.legend()
        # Move the legend to the right of the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Increase the space to the right of the plot to accommodate the legend
        plt.subplots_adjust(right=0.75)

        # Set y-axis limits from 0 to 1
        plt.ylim(0, 1)

        # Save the plot to the specified file
        plt.savefig(f"aws/plots/{prefix}{goal}_{i}.png", bbox_inches="tight")
        print(f"Saved {prefix}{goal}_{i}.png")
        plt.clf()  # Clear the current figure after saving the plot

def drawaverage():
    # goal0_data = np.array([d for d in data if d[4] == 0 and d[0] % 100 == 0], dtype=float)
    # goal1_data = np.array([d for d in data if d[4] == 1 and d[0] % 100 == 0], dtype=float)
    datas = [data_map[i + 1] for i in range(10)]

    gen_data: dict[int, list] = {}
    for run_data in datas:
        for d in run_data:
            gen = d[0]
            if gen not in gen_data:
                gen_data[gen] = []
            gen_data[gen].append(d)
    sorted_gens = sorted(list(gen_data.keys()))
                
    avg0s = [
        [gen, np.average([d[1] for d in gen_data[gen] if d[4] == 0]), np.average([d[2] for d in gen_data[gen] if d[4] == 0])]
        for gen in sorted_gens
        if any(d[4] == 0 for d in gen_data[gen])
    ]
    if CHANGE_GOAL:
        avg1s = [
            [gen, np.average([d[1] for d in gen_data[gen] if d[4] == 1]), np.average([d[2] for d in gen_data[gen] if d[4] == 1])]
            for gen in sorted_gens
            if any(d[4] == 1 for d in gen_data[gen])
        ]
    if BETA > 1:
        maleables = [
            [gen, np.average([d[3] for d in gen_data[gen]])]
            for gen in sorted_gens
        ]

    goal0_data = np.array(avg0s)
    if CHANGE_GOAL:
        goal1_data = np.array(avg1s)
    if BETA > 1:
        maleable_data = np.array(maleables)

    window_size = 20  # Adjust this value for the desired moving average window size

    if BETA > 1:
        plt.plot(goal0_data[window_size - 1:, 0], moving_average(goal0_data[:, 1], window_size), label='Baldwin G1', color='#ff000080')
    plt.plot(goal0_data[window_size - 1:, 0], moving_average(goal0_data[:, 2], window_size), label='G1', color='#ff000040' if BETA > 1 else "#ff000080")
    
    if CHANGE_GOAL:
        if BETA > 1:
            plt.plot(goal1_data[window_size - 1:, 0], moving_average(goal1_data[:, 1], window_size), label='Baldwin G2', color='#0000ff80')
        plt.plot(goal1_data[window_size - 1:, 0], moving_average(goal1_data[:, 2], window_size), label='G2', color='#0000ff40'if BETA > 1 else "#0000ff80")

    if BETA > 1:
        plt.plot(maleable_data[window_size - 1:, 0], moving_average(maleable_data[:, 1] / 100, window_size), label='Plasticity (Moving Avg)', color='#00aa00')


    plt.title('Average ' + TITLE)

    plt.xlabel('Generations')
    plt.ylabel('Fitness')

    plt.legend()
    # Move the legend to the right of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Increase the space to the right of the plot to accommodate the legend
    plt.subplots_adjust(right=0.75)

    # Set y-axis limits from 0 to 1
    plt.ylim(0, 1)

    # Save the plot to the specified file
    plt.savefig(f"aws/plots/{prefix}{goal}.png", bbox_inches="tight")
    print(f"Saved {prefix}{goal}.png")
    plt.clf()

drawall()
drawaverage()