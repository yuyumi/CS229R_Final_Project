import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python plot_all.py <beta> <change_goal> <?generations>")
    exit(1)

BETA = int(sys.argv[1])
CHANGE_GOAL = bool(int(sys.argv[2]))
GENERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

prefix = f'baldwin_{BETA}_' if BETA > 1 else ''
goal = 'fixed' if not CHANGE_GOAL else 'fvg'

header = ['Generation', 'G1 Baldwin Fitness', 'G1 Fitness', 'G2 Baldwin Fitness', 'G2 Fitness', '% Plasticity']

data = csv.reader(open(f"aws/data/{prefix}{goal}.csv", "r"))
fixed_header = next(data)
assert all(h1 == h2 for h1, h2 in zip(header, fixed_header))

data = [list(d) for d in data if float(d[0]) <= GENERATIONS]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


plt.plot([float(d[0]) for d in data], [float(d[1]) for d in data], label='Average', color='#0000ffff')
# plt.plot([float(d[0]) for d in data], [float(d[2]) for d in data], label='Average', color='#0000ffff')

for i in range(1, 11):
    indiv_data = csv.reader(open(f"aws/data/{prefix}{goal}_{i}.csv", "r"))
    indiv_header = next(indiv_data)
    assert all(h1 == h2 for h1, h2 in zip(header, indiv_header))

    indiv_data = [list(d) for d in indiv_data if float(d[0]) <= GENERATIONS]

    plt.plot([float(d[0]) for d in indiv_data], [float(d[1]) for d in indiv_data], label=f'Population {i}', color=f'#0000ff20')
    # plt.plot([float(d[0]) for d in indiv_data], [float(d[2]) for d in indiv_data], label=f'Population {i}', color=f'#0000ff20')

baldwin_portion = f"Baldwin (Beta = {BETA}) " if BETA > 1 else ""
plt.title(f'Average {"Fixed Goal" if not CHANGE_GOAL else "FVG"} {baldwin_portion}G1 Fitness', pad=20)

plt.xlabel('Generations')
plt.ylabel('Fitness')

plt.plot()

plt.legend()
# Move the legend to the right of the plot
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Hide the legend
plt.legend().set_visible(False)

# Increase the space to the right of the plot to accommodate the legend
plt.subplots_adjust(right=0.75)

# Set y-axis limits from 0 to 1
plt.ylim(0.6, 1)

# Save the plot to the specified file
plt.savefig(f"aws/plots/all_{prefix}{goal}.png", bbox_inches="tight")
plt.clf()
