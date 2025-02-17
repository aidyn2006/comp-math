import numpy as np
from tabulate import tabulate

n = int(input('Enter number of data points: '))

x = np.zeros(n)
y = np.zeros((n, n))

print('Enter data for x and y:')
for i in range(n):
    x[i] = float(input(f'x[{i}]= '))
    y[i][0] = float(input(f'y[{i}]= '))

for i in range(1, n):
    for j in range(n - i):
        y[j][i] = y[j + 1][i - 1] - y[j][i - 1]

table = []
for i in range(n):
    row = [f"{x[i]:.2f}", f"{y[i][0]:.2f}"]
    row += [f"{y[i][j]:.2f}" if j < n - i else "" for j in range(1, n)]
    table.append(row)

headers = ["x", "f(x)"] + [f"Δ^{i}y" for i in range(1, n)]

print(y[1][3])
print("\nFORWARD DIFFERENCE TABLE\n")
print(tabulate(table, headers=headers, tablefmt="grid"))