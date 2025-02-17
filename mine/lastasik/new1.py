import numpy as np

# 1. Trapezoidal Rule for ∫(0 to 1) x^3 dx with 5 sub-intervals
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])

# Function for x^3
f1 = lambda x: x ** 3
result1 = trapezoidal_rule(f1, 0, 1, 5)

# 2. Simpson's 1/3 Rule
def simpsons_one_third_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1  # Ensure even number of intervals
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2]) + y[n])

# (i) ∫(0 to π) sin(x) dx with 10 intervals (11 ordinates)
f2 = np.sin
result2_i = simpsons_one_third_rule(f2, 0, np.pi, 10)

# (ii) ∫(0 to π/2) √(cos(θ)) dθ with 8 intervals (9 ordinates)
f3 = lambda theta: np.sqrt(np.cos(theta))
result2_ii = simpsons_one_third_rule(f3, 0, np.pi/2, 8)

# 3. Simpson's 3/8 Rule
def simpsons_three_eighth_rule(f, a, b, n):
    if n % 3 != 0:
        n += 3 - (n % 3)  # Ensure n is a multiple of 3
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return (3 * h / 8) * (y[0] + 3 * np.sum(y[1:n:3] + y[2:n:3]) + 2 * np.sum(y[3:n-1:3]) + y[n])

# (i) ∫(0 to 9) dx / (1 + x^3)
f4 = lambda x: 1 / (1 + x ** 3)
result3_i = simpsons_three_eighth_rule(f4, 0, 9, 9)

# (ii) ∫(0 to π/2) sin(x) dx
result3_ii = simpsons_three_eighth_rule(np.sin, 0, np.pi / 2, 6)

# 4. Evaluate ∫(0 to 1) dx / (1 + x) using three methods
f5 = lambda x: 1 / (1 + x)

# (i) Trapezoidal Rule
result4_i = trapezoidal_rule(f5, 0, 1, 4)

# (ii) Simpson's 1/3 Rule
result4_ii = simpsons_one_third_rule(f5, 0, 1, 4)

# (iii) Simpson's 3/8 Rule
result4_iii = simpsons_three_eighth_rule(f5, 0, 1, 6)

# Output results
print("1. Trapezoidal Rule (∫₀¹ x³ dx):", result1)
print("2. Simpson's 1/3 Rule:")
print("   (i) ∫₀^π sin(x) dx:", result2_i)
print("   (ii) ∫₀^(π/2) √(cos(θ)) dθ:", result2_ii)
print("3. Simpson's 3/8 Rule:")
print("   (i) ∫₀⁹ dx / (1 + x³):", result3_i)
print("   (ii) ∫₀^(π/2) sin(x) dx:", result3_ii)
print("4. Evaluation of ∫₀¹ dx / (1 + x):")
print("   (i) Trapezoidal Rule:", result4_i)
print("   (ii) Simpson's 1/3 Rule:", result4_ii)
print("   (iii) Simpson's 3/8 Rule:", result4_iii)
