import math

import matplotlib.pyplot as plt

def func(x):
    return x-math.cos(x)

def bisec(a, b, E, iterate=1000):
    if func(a) * func(b) >= 0:
        print("The function must have opposite signs")
        return None

    x_prev = a
    x_values = []
    error_values = []
    iterations = []

    for i in range(iterate):
        mid = (a + b) / 2
        x_values.append(mid)
        iterations.append(i + 1)

        error = abs(mid - x_prev)
        error_values.append(error)

        if error < E:
            plot_convergence(iterations, x_values, error_values)
            print(f"Root found after {i + 1} iterations: {mid}")
            return mid

        if func(mid) * func(a) < 0:
            b = mid
        else:
            a = mid

        x_prev = mid

    plot_convergence(iterations, x_values, error_values)
    print(f"Maximum iterations reached. Root approximation: {mid}")
    return mid

def plot_convergence(iterations, x_values, error_values):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, x_values, marker='o', label='xj values')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Midpoint (xj)")
    plt.title("Convergence of the Bisection Method")
    plt.grid(color="hotpink", linestyle='--')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, error_values, marker='o', color='r', label='Error')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.title("Error Reduction in Bisection Method")
    plt.yscale('log')
    plt.grid(color="hotpink", linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.show()

a = float(input("Enter a: "))
b = float(input("Enter b: "))
epsilon = float(input("Enter epsilon: "))

root = bisec(a, b, epsilon)
