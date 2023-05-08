import re
import sys
import numpy as np
import csv

if len(sys.argv) < 3:
    print("Usage: python plotter.py <beta> <change_goal>")
    exit(1)

BETA = int(sys.argv[1])
CHANGE_GOAL = bool(int(sys.argv[2]))

prefix = f'baldwin_{BETA}_' if BETA > 1 else ''
goal = f'fvg' if CHANGE_GOAL else 'fixed'

# Format:
format = re.compile(
    r'\[Generation (\d+)\] Fitness: (\d+\.\d+) No-Baldwin Fitness: (\d+\.\d+) Maleable: (\d+\.\d+)% Goal: (\d) Times: (\d+\.\d+)s')

header = ['Generation', 'G1 Baldwin Fitness', 'G1 Fitness', 'G2 Baldwin Fitness', 'G2 Fitness', '% Plasticity']

def main():
    data_map: dict[int, list] = {}
    data0_map = {}
    data1_map = {}

    min_len = None

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

        data = [d for d in data if d[0] % 100 == 0]
        min_len = len(data) if min_len is None else min(min_len, len(data))

        # Only where Goal is 0
        goal0_data = []
        goal1_data = []
        for j, d in enumerate(data):
            if d[4] == 0:
                while len(goal0_data) <= j:
                    goal0_data.append([d[1], d[2]])
            elif len(goal0_data) > 0:
                goal0_data.append(goal0_data[-1])

            if d[4] == 1:
                while len(goal1_data) <= j:
                    goal1_data.append([d[1], d[2]])
            elif len(goal1_data) > 0:
                goal1_data.append(goal1_data[-1])

        data0_map[i] = goal0_data
        data1_map[i] = goal1_data

        data_map[i] = data

    for i in range(1, 11):
        with open(f'aws/data/{prefix}{goal}_{i}.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for g in range(min_len):
                gen = data_map[i][g][0]
                g1_baldwin = data0_map[i][g][0]
                g1 = data0_map[i][g][1]
                g2_baldwin = None if not CHANGE_GOAL else data1_map[i][g][0]
                g2 = None if not CHANGE_GOAL else data1_map[i][g][1]
                plasticity = data_map[i][g][3] / 100

                row = [gen, g1_baldwin, g1, g2_baldwin, g2, plasticity]
                writer.writerow(row)

    with open(f'aws/data/{prefix}{goal}.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(min_len):
            gen = data_map[1][i][0]
            g1_baldwin = np.mean([data0_map[j][i][0] for j in range(1, 11)])
            g1 = np.mean([data0_map[j][i][1] for j in range(1, 11)])
            g2_baldwin = None if not CHANGE_GOAL else np.mean([data1_map[j][i][0] for j in range(1, 11)])
            g2 = None if not CHANGE_GOAL else np.mean([data1_map[j][i][1] for j in range(1, 11)])
            plasticity = np.mean([data_map[j][i][3] for j in range(1, 11)]) / 100

            row = [gen, g1_baldwin, g1, g2_baldwin, g2, plasticity]
            writer.writerow(row)

if __name__ == '__main__':
    main()