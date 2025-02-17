import numpy as np
from tabulate import tabulate

n = int(input('Enter number of data points: '))

x = np.zeros(n)
y = np.zeros((n, n))

print('Enter data for x and f(x):')
for i in range(n):
    x[i] = float(input(f'x[{i}]= '))
    y[i][0] = float(input(f'f(x)[{i}]= '))

h = x[1] - x[0]

for i in range(1, n-1):
    y[i][1] = (y[i+1][0] - y[i-1][0]) / (2*h)

for j in range(2, n):
    for i in range(1, n - j):
        y[i][j] = y[i+1][j-1] - y[i][j-1]

table = []
for i in range(n):
    row = [f"{x[i]:.2f}", f"{y[i][0]:.2f}"]
    row += [f"{y[i][j]:.2f}" if j < n - i else "" for j in range(1, n)]
    table.append(row)

headers = ["x", "f(x)"] + [f"δ^{i}y" for i in range(1, n)]

print("\nCENTRAL DIFFERENCE TABLE\n")
print(tabulate(table, headers=headers, tablefmt="grid"))