import numpy as np
from scipy.integrate import quad

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        result += 2 * f(a + i * h)

    return (h / 2) * result


def simpsons_one_third_rule(f, a, b, n):
    if n % 2 == 1:
        n += 1  # n must be even
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        if i % 2 == 1:
            result += 4 * f(a + i * h)
        else:
            result += 2 * f(a + i * h)

    return (h / 3) * result


def simpsons_three_eighth_rule(f, a, b, n):
    if n % 3 != 0:
        n += 3 + (n % 3)  # n must be a multiple of 3
    h = (b - a) / n
    result = f(a) + f(b)

    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * f(a + i * h)
        else:
            result += 3 * f(a + i * h)

    return (3 * h / 8) * result


# Problem 1: ∫₀¹ x³ dx using trapezoidal rule with 5 sub-intervals
def problem1():
    f = lambda x: x ** 3
    result = trapezoidal_rule(f, 0, 1, 5)
    print(f"Problem 1 - Trapezoidal Rule (∫₀¹ x³ dx):")
    print(f"Result: {result}")
    print(f"Actual value: {0.25}")  # Analytical solution is 1/4
    print(f"Error: {abs(0.25 - result)}\n")


# Problem 2: Simpson's 1/3 rule
def problem2():
    # (i) ∫₀π sin x dx with 11 ordinates
    f1 = lambda x: np.sin(x)
    result1 = simpsons_one_third_rule(f1, 0, np.pi, 10)  # 11 ordinates means 10 intervals

    # (ii) ∫₀π/2 √cos θ dθ with 9 ordinates
    f2 = lambda x: np.sqrt(np.cos(x))
    result2 = simpsons_one_third_rule(f2, 0, np.pi / 2, 8)  # 9 ordinates means 8 intervals

    print("Problem 2 - Simpson's 1/3 Rule:")
    print(f"(i) ∫₀π sin x dx = {result1}")
    print(f"    Actual value: {2.0}")  # Analytical solution is 2
    print(f"    Error: {abs(2.0 - result1)}")
    print(f"(ii) ∫₀π/2 √cos θ dθ = {result2}\n")


# Problem 3: Simpson's 3/8 rule
def problem3():
    # (i) ∫₀9 dx/(1+x³)
    f1 = lambda x: 1 / (1 + x ** 3)
    result1 = simpsons_three_eighth_rule(f1, 0, 9, 9)

    # (ii) ∫₀π/2 sin x dx
    f2 = lambda x: np.sin(x)
    result2 = simpsons_three_eighth_rule(f2, 0, np.pi / 2, 9)

    print("Problem 3 - Simpson's 3/8 Rule:")
    print(f"(i) ∫₀9 dx/(1+x³) = {result1}")
    print(f"(ii) ∫₀π/2 sin x dx = {result2}")
    print(f"     Actual value for (ii): {1.0}")  # Analytical solution is 1
    print(f"     Error for (ii): {abs(1.0 - result2)}\n")


# Problem 4: Compare all methods
def problem4():
    f = lambda x: 1 / (1 + x)

    # (i) Trapezoidal rule
    trap_result = trapezoidal_rule(f, 0, 1, 8)

    # (ii) Simpson's 1/3 rule
    simp13_result = simpsons_one_third_rule(f, 0, 1, 8)

    # (iii) Simpson's 3/8 rule
    simp38_result = simpsons_three_eighth_rule(f, 0, 1, 9)

    print("Problem 4 - Evaluate ∫₀¹ dx/(1+x):")
    print("Actual value: 0.69314718...")  # ln(2)
    print(f"(i) Trapezoidal Rule: {trap_result}")
    print(f"    Error: {abs(np.log(2) - trap_result)}")
    print(f"(ii) Simpson's 1/3 Rule: {simp13_result}")
    print(f"     Error: {abs(np.log(2) - simp13_result)}")
    print(f"(iii) Simpson's 3/8 Rule: {simp38_result}")
    print(f"      Error: {abs(np.log(2) - simp38_result)}")


# Run all problems
if __name__ == "__main__":
    problem1()
    problem2()
    problem3()
    problem4()