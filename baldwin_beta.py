import csv
import numpy as np
import matplotlib.pyplot as plt

header = ['Generation', 'G1 Baldwin Fitness', 'G1 Fitness', 'G2 Baldwin Fitness', 'G2 Fitness', '% Plasticity']

betas = [1, 5, 10, 20]
prefixes = ['', 'baldwin_5_', 'baldwin_10_', 'baldwin_20_']

data = {}
for beta, prefix in zip(betas, prefixes):
    data_reader = csv.reader(open(f"aws/data/{prefix}fvg.csv", "r"))
    data_header = next(data_reader)
    assert all(h1 == h2 for h1, h2 in zip(header, data_header))

    data[beta] = [list(d) for d in data_reader if float(list(d)[0]) <= 10000]


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def exponential_moving_average(data, window_size):
    alpha = 2 / (window_size + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

WS = 1

# plt.plot([float(d[0]) for d in data[1]], exponential_moving_average([float(d[2]) for d in data[1]], WS), label='Fitness (Beta = 1)', color='#ff0000ff')
plt.plot([float(d[0]) for d in data[1]], [float(d[5]) for d in data[1]], label='Plasticity (Beta = 1)', color='#00b010ff')

# plt.plot([float(d[0]) for d in data[5]], exponential_moving_average([float(d[2]) for d in data[5]], WS), label='Fitness (Beta = 5)', color='#ff0000cc')
plt.plot([float(d[0]) for d in data[5]], [float(d[5]) for d in data[5]], label='Plasticity (Beta = 5)', color='#00b010cc')

# plt.plot([float(d[0]) for d in data[10]], exponential_moving_average([float(d[2]) for d in data[10]], WS), label='Fitness (Beta = 10)', color='#ff000060')
plt.plot([float(d[0]) for d in data[10]], [float(d[5]) for d in data[10]], label='Plasticity (Beta = 10)', color='#00b01060')

# plt.plot([float(d[0]) for d in data[20]], exponential_moving_average([float(d[2]) for d in data[20]], WS), label='Fitness (Beta = 20)', color='#ff000030')
plt.plot([float(d[0]) for d in data[20]], [float(d[5]) for d in data[20]], label='Plasticity (Beta = 20)', color='#00b01030')

plt.title(f'Different Beta Values for Plasticity Over Generations', pad=20)

plt.xlabel('Generations')
plt.ylabel('Plasticity')
# plt.ylabel('Fitness')

plt.plot()

plt.legend()
# Move the legend to the right of the plot
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Increase the space to the right of the plot to accommodate the legend
plt.subplots_adjust(right=0.75)

# Set y-axis limits from 0 to 1
plt.ylim(0.1, 0.8)

# Save the plot to the specified file
plt.savefig(f"aws/plots/baldwin_beta_plasticity.png", bbox_inches="tight")
plt.clf()
