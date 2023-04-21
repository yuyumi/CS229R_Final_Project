import re
import matplotlib.pyplot as plt
import numpy as np

FILE = 'logs/baldwin_fixed_run1.txt'
with open(FILE, 'r') as f:
    data = f.read()

lines = data.split('\n')

# Format:
format = re.compile(r'\[Generation (\d+)\] Fitness: (\d+\.\d+) No-Baldwin Fitness: (\d+\.\d+) Maleable: (\d+\.\d+)% Goal: (\d) Times: (\d+\.\d+)s')
# format = re.compile(r'\[Generation (\d+)\] Fitness: (\d+\.\d+) Goal: (\d)')

header = ['Generation', 'Fitness', 'Pure Fitness', '% Maleable', 'Goal', 'Time']
# header = ['Generation', 'Fitness', 'Goal']

data = []
for line in lines:
    match = format.match(line)
    if match:
        data.append(tuple(map(float, match.groups())))

data = [d for d in data if d[0] <= 30000]

# Only where Goal is 0
# goal0_data = np.array([d for d in data if d[4] == 0 and d[0] % 100 == 0], dtype=float)
goal1_data = np.array([d for d in data if d[4] == 1 and d[0] % 100 == 0], dtype=float)
# goal0_data = np.array([d for d in data if d[2] == 0 and d[0] % 100 == 0], dtype=float)
# goal1_data = np.array([d for d in data if d[2] == 1 and d[0] % 100 == 0], dtype=float)


# Plot fitness over generations but separate series based on goal (either 0 or 1)
data = np.array(data, dtype=float)
# plt.plot(goal0_data[:, 0], goal0_data[:, 1], label='Baldwin G1', color='#ff000080')
# plt.plot(goal0_data[:, 0], goal0_data[:, 2], label='G1', color='#ff000040')
plt.plot(goal1_data[:, 0], goal1_data[:, 1], label='Baldwin G2', color='#0000ff80')
plt.plot(goal1_data[:, 0], goal1_data[:, 2], label='G2', color='#0000ff40')

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window_size = 10  # Adjust this value for the desired moving average window size
maleable_moving_avg = moving_average(data[:, 3] / 100, window_size)
plt.plot(data[window_size - 1:, 0], maleable_moving_avg, label='Plasticity (Moving Avg)', color='#00aa00')


plt.title("Fixed Goal Baldwin Fitness over Generations")

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
plt.savefig("plots/baldwin_fixed_run1.png", bbox_inches="tight")

# plt.show()
