import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python baldwin_fix_vs_fvg.py <beta>")
    exit(1)

BETA = int(sys.argv[1])

header = ['Generation', 'G1 Baldwin Fitness', 'G1 Fitness', 'G2 Baldwin Fitness', 'G2 Fitness', '% Plasticity']

data_fixed = csv.reader(open(f"aws/data/baldwin_{BETA}_fixed.csv", "r"))
fixed_header = next(data_fixed)
assert all(h1 == h2 for h1, h2 in zip(header, fixed_header))

data_fvg = csv.reader(open(f"aws/data/baldwin_{BETA}_fvg.csv", "r"))
fvg_header = next(data_fvg)
assert all(h1 == h2 for h1, h2 in zip(header, fvg_header))

data_fixed = [list(d) for d in data_fixed if float(d[0]) <= 10000]
data_fvg = [list(d) for d in data_fvg if float(d[0]) <= 10000]

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

plt.plot([float(d[0]) for d in data_fixed], [float(d[1]) for d in data_fixed], label='Fixed Goal Fitness', color='#ff000080')
plt.plot([float(d[0]) for d in data_fixed], [float(d[5]) for d in data_fixed], label='Fixed Goal Plasticity', color='#00b01080')

plt.plot([float(d[0]) for d in data_fvg], [float(d[1]) for d in data_fvg], label='MVG Fitness', color='#ff0000ff')
plt.plot([float(d[0]) for d in data_fvg], [float(d[5]) for d in data_fvg], label='MVG Plasticity', color='#00b010ff')

plt.title(f'Average Baldwin G1 Fitness (Beta = {BETA}) (MVG vs Fixed Goal)', pad=20)

plt.xlabel('Generations')
plt.ylabel('Fitness / Plasticity')

plt.plot()

plt.legend()
# Move the legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Increase the space to the right of the plot to accommodate the legend
plt.subplots_adjust(right=0.75)

# Set y-axis limits from 0 to 1
plt.ylim(0, 1)

# Save the plot to the specified file
plt.savefig(f"aws/plots/baldwin_{BETA}_fixed_vs_fvg.png", bbox_inches="tight")
plt.clf()
