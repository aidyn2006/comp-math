import numpy as np
import sympy as sp
import math

print("=== PART 1. NUMERICAL DIFFERENTIATION ===\n")

# ------------------------------
# Problem 1:
# Given: t: 0, 5, 10, 15, 20 (sec) and v: 0, 3, 14, 69, 228 (m/s)
# Find the initial acceleration (dv/dt at t=0) using all the data.
# For an equal step h = 5, we can use the five-point forward difference formula for the first derivative:
#     f'(x0) ≈ (-25 f0 + 48 f1 - 36 f2 + 16 f3 - 3 f4) / (12 h)
# (Reference: finite difference formulas)

t = np.array([0, 5, 10, 15, 20], dtype=float)
v = np.array([0, 3, 14, 69, 228], dtype=float)
h1 = 5.0

# Using the five-point forward formula at t = 0:
dvdt0 = (-25*v[0] + 48*v[1] - 36*v[2] + 16*v[3] - 3*v[4]) / (12 * h1)
print("Problem 1. Initial acceleration dv/dt (t=0) =", dvdt0, "m/s²")
# Expected result is approximately 1 m/s²

# ------------------------------
# Problem 2:
# Find f'(10) from the following data:
#    x:      3,   5,   11,    27,   34
#    f(x): -13,  23,  899, 17315, 35606
#
# The x-values are unequally spaced. To compute the derivative at x = 10 (which lies between 5 and 11)
# we construct the Newton's divided difference interpolation polynomial using sympy,
# differentiate it, and then evaluate at x = 10.

x_vals = np.array([3, 5, 11, 27, 34], dtype=float)
f_vals = np.array([-13, 23, 899, 17315, 35606], dtype=float)

# Define the symbol x and build the Newton interpolation polynomial
x = sp.symbols('x')
n = len(x_vals)
div_diff = np.zeros((n, n))
div_diff[:,0] = f_vals.copy()

for j in range(1, n):
    for i in range(n-j):
        div_diff[i,j] = (div_diff[i+1, j-1] - div_diff[i,j-1])/(x_vals[i+j] - x_vals[i])

# Build the Newton polynomial:
poly = div_diff[0,0]
term = 1
for j in range(1, n):
    term *= (x - x_vals[j-1])
    poly += div_diff[0,j]*term

poly_simpl = sp.simplify(poly)
print("\nProblem 2. Newton's interpolation polynomial:")
sp.pprint(poly_simpl)

# Now, differentiate the polynomial and evaluate at x=10:
poly_prime = sp.diff(poly_simpl, x)
fprime_at_10 = poly_prime.subs(x, 10)
print("\nf'(10) =", fprime_at_10)

# ------------------------------
# Problem 3:
# Given:
#    x:      1.5,   2.0,   2.5,   3.0,   3.5,   4.0
#    f(x): 3.375, 7.000, 13.625, 24.000, 38.875, 59.000
#
# Find the first, second, and third derivatives at x = 1.5.
# The points are equally spaced with h = 0.5, so we use the forward finite difference formulas:
# For f'(x0) at u = 0:
#     f'(x0) ≈ (1/h)[ Δf0 - (1/2) Δ²f0 + (1/3) Δ³f0 ]
# For f''(x0):
#     f''(x0) ≈ (1/h²)[ Δ²f0 - Δ³f0 ]
# For f'''(x0):
#     f'''(x0) ≈ (1/h³)[ Δ³f0 ]
#
# We build the difference table first.

x3 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
f3 = np.array([3.375, 7.000, 13.625, 24.000, 38.875, 59.000], dtype=float)
h3 = 0.5

# First differences:
delta1 = np.diff(f3)
# Second differences:
delta2 = np.diff(delta1)
# Third differences:
delta3 = np.diff(delta2)
# Fourth differences (not needed, but for reference):
delta4 = np.diff(delta3)

print("\nProblem 3. Finite difference table:")
print("f:", f3)
print("Δf:", delta1)
print("Δ²f:", delta2)
print("Δ³f:", delta3)

f1_at_1_5 = (1/h3)*(delta1[0] - 0.5*delta2[0] + (1/3)*delta3[0])
f2_at_1_5 = (1/(h3**2))*(delta2[0] - delta3[0])
f3_at_1_5 = (1/(h3**3))*(delta3[0])
print("\nDerivatives at x = 1.5:")
print("f'(1.5) =", f1_at_1_5)
print("f''(1.5) =", f2_at_1_5)
print("f'''(1.5) =", f3_at_1_5)

# ------------------------------
# Problem 4:
# Given:
#    x:     1.0,  1.2,  1.4,  1.6,  1.8,  2.0
#    f(x):  0,   0.128, 0.544, 1.296, 2.432, 4.00
#
# Find the first and second derivatives at x = 1.1.
# Note that x = 1.1 is not one of the tabulated points.
# We construct the finite difference table starting from x0 = 1.0 with step h = 0.2,
# and use Newton's forward formula with u = (x - x0)/h.
# Here, u = (1.1 - 1.0)/0.2 = 0.5

x4 = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=float)
f4 = np.array([0, 0.128, 0.544, 1.296, 2.432, 4.00], dtype=float)
h4 = 0.2
u = (1.1 - 1.0) / h4  # u = 0.5

# Compute finite differences:
delta1_4 = np.diff(f4)
delta2_4 = np.diff(delta1_4)
delta3_4 = np.diff(delta2_4)
delta4_4 = np.diff(delta3_4)

# Newton's forward interpolation formula is:
#   f(x0 + u*h) = f0 + u Δf0 + [u(u-1)/2!] Δ²f0 + [u(u-1)(u-2)/3!] Δ³f0 + [u(u-1)(u-2)(u-3)/4!] Δ⁴f0
#
# Then the derivative with respect to x is: f'(x) = (1/h) * d/du[...]
# The derivative d/du of the expansion is:
#   Δf0 + ((2*u - 1)/2) Δ²f0 + ((3*u^2 - 6*u + 2)/6) Δ³f0 + ...
#
fprime_1_1 = (1/h4) * (delta1_4[0] + ((2*u - 1)/2)*delta2_4[0] + ((3*u**2 - 6*u + 2)/6)*delta3_4[0])
print("\nProblem 4. Derivatives at x = 1.1:")
print("f'(1.1) ≈", fprime_1_1)

# For the second derivative:
# f''(x) = (1/h²)*[ d²/d(u²) (Newton expansion) ]
# Note: d²/du²: for uΔf0 → 0, for [u(u-1)/2] → 1, for [u(u-1)(u-2)/6] → (u-1)
fsecond_1_1 = (1/(h4**2)) * ( delta2_4[0] + (u - 1)*delta3_4[0] )
print("f''(1.1) ≈", fsecond_1_1)

# ------------------------------
# Problem 5:
# Given the following table:
#    x:  1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30
#    y:  1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140
#
# Find dy/dx and d²y/dx² at:
# (a) x = 1.05, (b) x = 1.25, (c) x = 1.15.
# The step size h = 0.05.
# Use forward/backward formulas at the boundaries and central differences for interior points.

x5 = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30], dtype=float)
y5 = np.array([1.000, 1.025, 1.049, 1.072, 1.095, 1.118, 1.140], dtype=float)
h5 = 0.05

# (a) At x = 1.05 (second point, use forward differences)
# First derivative:
#   f'(x0) ≈ (-3f0 + 4f1 - f2) / (2h)
fprime_1_05 = (-3*y5[0] + 4*y5[1] - y5[2]) / (2*h5)
# Second derivative:
#   f''(x0) ≈ (f0 - 2f1 + f2) / h²
fsecond_1_05 = (y5[0] - 2*y5[1] + y5[2]) / (h5**2)
print("\nProblem 5 (a) at x = 1.05:")
print("dy/dx ≈", fprime_1_05)
print("d²y/dx² ≈", fsecond_1_05)

# (b) At x = 1.25 (second-to-last point, use backward differences)
# First derivative:
#   f'(x) ≈ (3f_n - 4f_{n-1} + f_{n-2}) / (2h)
fprime_1_25 = (3*y5[-2] - 4*y5[-3] + y5[-4]) / (2*h5)
# Second derivative:
#   f''(x) ≈ (f_n - 2f_{n-1} + f_{n-2}) / h²
fsecond_1_25 = (y5[-1] - 2*y5[-2] + y5[-3]) / (h5**2)
print("\nProblem 5 (b) at x = 1.25:")
print("dy/dx ≈", fprime_1_25)
print("d²y/dx² ≈", fsecond_1_25)

# (c) At x = 1.15 (an interior point, use central differences)
# Here, x=1.15 is the 4th point (index 3 if indexing from 0: 1.00, 1.05, 1.10, 1.15,...)
# First derivative:
#   f'(x) ≈ (f(x+h) - f(x-h))/(2h)
idx = 3  # x = 1.15
fprime_1_15 = (y5[idx+1] - y5[idx-1]) / (2*h5)
# Second derivative:
#   f''(x) ≈ (f(x+h) - 2f(x) + f(x-h))/(h²)
fsecond_1_15 = (y5[idx+1] - 2*y5[idx] + y5[idx-1]) / (h5**2)
print("\nProblem 5 (c) at x = 1.15:")
print("dy/dx ≈", fprime_1_15)
print("d²y/dx² ≈", fsecond_1_15)

print("\n=== PART 2. NUMERICAL INTEGRATION ===\n")

# We now use composite rules for integration.

# ------------------------------
# Problem 1:
# Evaluate I = ∫₀¹ x³ dx using the Trapezoidal rule with 5 sub-intervals.
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
# (i) Evaluate I = ∫₀^π sin x dx using Simpson’s 1/3 rule,
#     taking 11 ordinates (thus n = 10 sub-intervals, which is even).
a2i, b2i = 0.0, math.pi
n2i = 10  # number of sub-intervals (must be even)
h2i = (b2i - a2i) / n2i
x_nodes2i = np.linspace(a2i, b2i, n2i+1)
f_nodes2i = np.sin(x_nodes2i)

simpson1_3_i = (h2i/3) * (f_nodes2i[0] + f_nodes2i[-1] +
                           4*np.sum(f_nodes2i[1:-1:2]) +
                           2*np.sum(f_nodes2i[2:-1:2]))
print("\nProblem 2(i). ∫₀^π sin x dx by Simpson’s 1/3 rule =", simpson1_3_i)
print("Exact value =", 2.0)

# (ii) Evaluate I = ∫₀^(π/2) √(cos θ) dθ using Simpson’s 1/3 rule,
#      taking 9 ordinates (thus n = 8 sub-intervals).
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
# Evaluate using Simpson’s 3/8 rule:
# (i) I = ∫₀^9 dx/(1+x³)
def f3i(x):
    return 1/(1+x**3)
a3i, b3i = 0.0, 9.0
# For Simpson’s 3/8 rule, the number of sub-intervals must be a multiple of 3; we choose n = 9.
n3i = 9
h3i = (b3i - a3i) / n3i
x_nodes3i = np.linspace(a3i, b3i, n3i+1)
f_nodes3i = f3i(x_nodes3i)

# Composite Simpson’s 3/8 rule:
simpson3_8_i = 0
# Break the interval into groups of 3 sub-intervals:
for i in range(0, n3i, 3):
    simpson3_8_i += (3*h3i/8)*(f_nodes3i[i] + 3*f_nodes3i[i+1] + 3*f_nodes3i[i+2] + f_nodes3i[i+3])
print("\nProblem 3(i). ∫₀^9 1/(1+x³) dx by Simpson’s 3/8 rule =", simpson3_8_i)

# (ii) I = ∫₀^(π/2) sin x dx using Simpson’s 3/8 rule.
a3ii, b3ii = 0.0, math.pi/2
n3ii = 3  # must be a multiple of 3
h3ii = (b3ii - a3ii) / n3ii
x_nodes3ii = np.linspace(a3ii, b3ii, n3ii+1)
f_nodes3ii = np.sin(x_nodes3ii)

simpson3_8_ii = (3*h3ii/8)*(f_nodes3ii[0] + 3*f_nodes3ii[1] + 3*f_nodes3ii[2] + f_nodes3ii[3])
print("Problem 3(ii). ∫₀^(π/2) sin x dx by Simpson’s 3/8 rule =", simpson3_8_ii)
print("Exact value =", 1.0)

# ------------------------------
# Problem 4:
# Evaluate I = ∫₀¹ dx/(1+x) by applying:
# (i) Trapezoidal rule, (ii) Simpson’s 1/3 rule, (iii) Simpson’s 3/8 rule.
def f4_int(x):
    return 1/(1+x)
a4, b4 = 0.0, 1.0

# For clarity, we choose n = 4 for (i) and (ii) (n must be even for Simpson’s 1/3 rule)
n4 = 4
h4_int = (b4 - a4) / n4
x_nodes4 = np.linspace(a4, b4, n4+1)
f_nodes4 = f4_int(x_nodes4)

# (i) Trapezoidal rule:
trapz4 = (h4_int/2) * (f_nodes4[0] + 2*np.sum(f_nodes4[1:-1]) + f_nodes4[-1])
print("\nProblem 4(i). ∫₀¹ 1/(1+x) dx by the Trapezoidal rule =", trapz4)

# (ii) Simpson’s 1/3 rule:
simpson1_3_4 = (h4_int/3) * (f_nodes4[0] + f_nodes4[-1] +
                              4*np.sum(f_nodes4[1:-1:2]) +
                              2*np.sum(f_nodes4[2:-1:2]))
print("Problem 4(ii). ∫₀¹ 1/(1+x) dx by Simpson’s 1/3 rule =", simpson1_3_4)

# (iii) For Simpson’s 3/8 rule, choose n = 3 (n must be a multiple of 3)
n4_38 = 3
h4_38 = (b4 - a4) / n4_38
x_nodes4_38 = np.linspace(a4, b4, n4_38+1)
f_nodes4_38 = f4_int(x_nodes4_38)
simpson3_8_4 = (3*h4_38/8)*(f_nodes4_38[0] + 3*f_nodes4_38[1] + 3*f_nodes4_38[2] + f_nodes4_38[3])
print("Problem 4(iii). ∫₀¹ 1/(1+x) dx by Simpson’s 3/8 rule =", simpson3_8_4)

# For reference, the exact value:
exact_val = np.log(2)
print("Exact value of ∫₀¹ 1/(1+x) dx =", exact_val)
