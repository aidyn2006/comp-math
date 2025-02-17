from matplotlib import pyplot as plt
from math import *

equation = input("Enter equation: ")
tolerance = float(input("Enter tolerance: "))

x_values = []
y_values = []
epsilon_values = []


def f(x):
    return eval(equation)


def find_bisection_interval(f, start=-10, end=10, step=0.1):
    x = start
    while x <= end:
        if f(x) * f(x + step) < 0:
            return x, x + step
        x += step
    return find_bisection_interval(f, start=start - 10, end=end + 10, step=step)


x1, x2 = 0.5, 2

count = 1

while x2 - x1 > tolerance:
    mid = (x1 + x2) / 2
    print(mid)
    x_values.append(mid)
    y_values.append(count)
    count += 1
    epsilon = abs(x2 - x1)
    epsilon_values.append(epsilon)

    if f(mid) == 0:
        break
    elif f(mid) * f(x1) < 0:
        x2 = mid
    else:
        x1 = mid

answer = (x2 + x1) / 2

print("The root of the equation is approximately:", round(answer, 3))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y_values, x_values, marker='o', label='xj values')
plt.xlabel("Number of Iterations")
plt.ylabel("Midpoint (xj)")
plt.title("Convergence of the Bisection Method")
plt.grid(color="hotpink", linestyle='--')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(y_values, epsilon_values, marker='o', color='r', label='Error')
plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.title("Error Reduction in Bisection Method")
plt.yscale('log')
plt.grid(color="hotpink", linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()
