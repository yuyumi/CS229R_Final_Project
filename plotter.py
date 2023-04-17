import re
import matplotlib.pyplot as plt
import numpy as np

FILE = 'logs/log_2023-04-12_23-19-35.txt'
with open(FILE, 'r') as f:
    data = f.read()

lines = data.split('\n')

# Format:
format = re.compile(r"\[Generation (\d+)\] Fitness: (\d+\.\d+) Goal: (\d)")
data = []
for line in lines:
    match = format.match(line)
    if match:
        data.append(tuple(map(float, match.groups())))

# Only where Goal is 0
goal0_data = np.array([d for d in data if d[2] == 0 and d[0] % 100 == 0], dtype=float)
goal1_data = np.array([d for d in data if d[2] == 1 and d[0] % 100 == 0], dtype=float)

# Plot fitness over generations but separate series based on goal (either 0 or 1)
data = np.array(data, dtype=float)
plt.plot(goal0_data[:, 0], goal0_data[:, 1], label='G1', color='#ff000080')
plt.plot(goal1_data[:, 0], goal1_data[:, 1], label='G2', color='#0000ff80')

plt.title("MVG Fitness over Generations")

plt.xlabel('Generations')
plt.ylabel('Fitness')

plt.legend()
plt.show()
