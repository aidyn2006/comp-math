import numpy as np
import sympy as sp
import math

print("\n=== PART 2. NUMERICAL INTEGRATION ===\n")


# ------------------------------
# Problem 1:
a, b = 0.0, 1.0
n1_int = 5
h_int1 = (b - a) / n1_int
x_nodes1 = np.linspace(a, b, n1_int+1)
f_nodes1 = x_nodes1**3

trapz1 = (h_int1/2) * (f_nodes1[0] + 2*np.sum(f_nodes1[1:-1]) + f_nodes1[-1])
print("Problem 1. ∫₀¹ x³ dx by the Trapezoidal rule =", trapz1)
print("Exact value =", 1/4)

# ------------------------------
# Problem 2:

a2i, b2i = 0.0, math.pi
n2i = 10
h2i = (b2i - a2i) / n2i
x_nodes2i = np.linspace(a2i, b2i, n2i+1)
f_nodes2i = np.sin(x_nodes2i)

simpson1_3_i = (h2i/3) * (f_nodes2i[0] + f_nodes2i[-1] +
                           4*np.sum(f_nodes2i[1:-1:2]) +
                           2*np.sum(f_nodes2i[2:-1:2]))
print("\nProblem 2(i). ∫₀^π sin x dx by Simpson’s 1/3 rule =", simpson1_3_i)
print("Exact value =", 2.0)

# (ii)
a2ii, b2ii = 0.0, math.pi/2
n2ii = 8
h2ii = (b2ii - a2ii) / n2ii
x_nodes2ii = np.linspace(a2ii, b2ii, n2ii+1)
f_nodes2ii = np.sqrt(np.cos(x_nodes2ii))

simpson1_3_ii = (h2ii/3) * (f_nodes2ii[0] + f_nodes2ii[-1] +
                             4*np.sum(f_nodes2ii[1:-1:2]) +
                             2*np.sum(f_nodes2ii[2:-1:2]))
print("\nProblem 2(ii). ∫₀^(π/2) √(cosθ) dθ by Simpson’s 1/3 rule =", simpson1_3_ii)

# ------------------------------
# Problem 3:
def f3i(x):
    return 1/(1+x**3)
a3i, b3i = 0.0, 9.0
n3i = 9
h3i = (b3i - a3i) / n3i
x_nodes3i = np.linspace(a3i, b3i, n3i+1)
f_nodes3i = f3i(x_nodes3i)

simpson3_8_i = 0
for i in range(0, n3i, 3):
    simpson3_8_i += (3*h3i/8)*(f_nodes3i[i] + 3*f_nodes3i[i+1] + 3*f_nodes3i[i+2] + f_nodes3i[i+3])
print("\nProblem 3(i). ∫₀^9 1/(1+x³) dx by Simpson’s 3/8 rule =", simpson3_8_i)

# (ii)
a3ii, b3ii = 0.0, math.pi/2
n3ii = 3
h3ii = (b3ii - a3ii) / n3ii
x_nodes3ii = np.linspace(a3ii, b3ii, n3ii+1)
f_nodes3ii = np.sin(x_nodes3ii)

simpson3_8_ii = (3*h3ii/8)*(f_nodes3ii[0] + 3*f_nodes3ii[1] + 3*f_nodes3ii[2] + f_nodes3ii[3])
print("Problem 3(ii). ∫₀^(π/2) sin x dx by Simpson’s 3/8 rule =", simpson3_8_ii)
print("Exact value =", 1.0)

# ------------------------------
# Problem 4:
def f4_int(x):
    return 1/(1+x)
a4, b4 = 0.0, 1.0

n4 = 4
h4_int = (b4 - a4) / n4
x_nodes4 = np.linspace(a4, b4, n4+1)
f_nodes4 = f4_int(x_nodes4)

# (i)
trapz4 = (h4_int/2) * (f_nodes4[0] + 2*np.sum(f_nodes4[1:-1]) + f_nodes4[-1])
print("\nProblem 4(i). ∫₀¹ 1/(1+x) dx by the Trapezoidal rule =", trapz4)

# (ii)
simpson1_3_4 = (h4_int/3) * (f_nodes4[0] + f_nodes4[-1] +
                              4*np.sum(f_nodes4[1:-1:2]) +
                              2*np.sum(f_nodes4[2:-1:2]))
print("Problem 4(ii). ∫₀¹ 1/(1+x) dx by Simpson’s 1/3 rule =", simpson1_3_4)

# (iii)
n4_38 = 3
h4_38 = (b4 - a4) / n4_38
x_nodes4_38 = np.linspace(a4, b4, n4_38+1)
f_nodes4_38 = f4_int(x_nodes4_38)
simpson3_8_4 = (3*h4_38/8)*(f_nodes4_38[0] + 3*f_nodes4_38[1] + 3*f_nodes4_38[2] + f_nodes4_38[3])
print("Problem 4(iii). ∫₀¹ 1/(1+x) dx by Simpson’s 3/8 rule =", simpson3_8_4)

exact_val = np.log(2)
print("Exact value of ∫₀¹ 1/(1+x) dx =", exact_val)
