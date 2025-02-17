import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp

# ------------------------------
# 1. Метод трапеций для интеграла ∫₀¹ x³ dx
a, b = 0.0, 1.0
n1 = 5
h1 = (b - a) / n1
x_nodes1 = np.linspace(a, b, n1 + 1)
f_nodes1 = x_nodes1 ** 3

# Вычисление интеграла методом трапеций
trapz1 = (h1 / 2) * (f_nodes1[0] + 2 * np.sum(f_nodes1[1:-1]) + f_nodes1[-1])
print("Problem 1. ∫₀¹ x³ dx by the Trapezoidal rule =", trapz1)
print("Exact value =", 1 / 4)

# Визуализация:
x_dense = np.linspace(a, b, 400)
y_dense = x_dense ** 3

plt.figure(figsize=(8, 5))
plt.plot(x_dense, y_dense, 'b-', label='$x^3$')
plt.plot(x_nodes1, f_nodes1, 'ro', label='Узлы')

# Рисуем трапеции
for i in range(n1):
    xs = [x_nodes1[i], x_nodes1[i], x_nodes1[i + 1], x_nodes1[i + 1]]
    ys = [0, f_nodes1[i], f_nodes1[i + 1], 0]
    plt.fill(xs, ys, 'r', edgecolor='k', alpha=0.3)

plt.title("Метод трапеций для $x^3$")
plt.xlabel("x")
plt.ylabel("$x^3$")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 2. Правило Симпсона 1/3

## (i) для ∫₀^π sin x dx
a2i, b2i = 0.0, math.pi
n2i = 10  # должно быть четным
h2i = (b2i - a2i) / n2i
x_nodes2i = np.linspace(a2i, b2i, n2i + 1)
f_nodes2i = np.sin(x_nodes2i)

simpson1_3_i = (h2i / 3) * (f_nodes2i[0] + f_nodes2i[-1] +
                            4 * np.sum(f_nodes2i[1:-1:2]) +
                            2 * np.sum(f_nodes2i[2:-1:2]))
print("\nProblem 2(i). ∫₀^π sin x dx by Simpson’s 1/3 rule =", simpson1_3_i)
print("Exact value =", 2.0)

# Визуализация: для каждого отрезка берём группы по 3 узлам и аппроксимируем параболой.
plt.figure(figsize=(8, 5))
x_dense2 = np.linspace(a2i, b2i, 400)
y_dense2 = np.sin(x_dense2)
plt.plot(x_dense2, y_dense2, 'b-', label='$\sin(x)$')
plt.plot(x_nodes2i, f_nodes2i, 'ro', label='Узлы')

# Для каждого набора из 2 отрезков (3 узла) аппроксимируем параболой
for i in range(0, n2i, 2):
    # Берём 3 узла: (x[i], f(x[i])), (x[i+1], f(x[i+1])), (x[i+2], f(x[i+2]))
    xi = x_nodes2i[i:i + 3]
    yi = f_nodes2i[i:i + 3]
    # Находим коэффициенты параболы: ax^2 + bx + c
    coeffs = np.polyfit(xi, yi, 2)
    poly = np.poly1d(coeffs)
    x_parab = np.linspace(xi[0], xi[-1], 100)
    plt.plot(x_parab, poly(x_parab), 'g--', linewidth=2, label='Параболическая аппроксимация' if i == 0 else "")
    # Закрашиваем область под параболой
    plt.fill_between(x_parab, poly(x_parab), color='green', alpha=0.2)

plt.title("Правило Симпсона 1/3 для $\sin(x)$")
plt.xlabel("x")
plt.ylabel("$\sin(x)$")
plt.legend()
plt.grid(True)
plt.show()

## (ii) для ∫₀^(π/2) √(cosθ) dθ
a2ii, b2ii = 0.0, math.pi / 2
n2ii = 8  # должно быть четным
h2ii = (b2ii - a2ii) / n2ii
x_nodes2ii = np.linspace(a2ii, b2ii, n2ii + 1)
f_nodes2ii = np.sqrt(np.cos(x_nodes2ii))

simpson1_3_ii = (h2ii / 3) * (f_nodes2ii[0] + f_nodes2ii[-1] +
                              4 * np.sum(f_nodes2ii[1:-1:2]) +
                              2 * np.sum(f_nodes2ii[2:-1:2]))
print("\nProblem 2(ii). ∫₀^(π/2) √(cosθ) dθ by Simpson’s 1/3 rule =", simpson1_3_ii)

# Визуализация:
plt.figure(figsize=(8, 5))
x_dense2ii = np.linspace(a2ii, b2ii, 400)
y_dense2ii = np.sqrt(np.cos(x_dense2ii))
plt.plot(x_dense2ii, y_dense2ii, 'b-', label='$\sqrt{\\cos(\\theta)}$')
plt.plot(x_nodes2ii, f_nodes2ii, 'ro', label='Узлы')

for i in range(0, n2ii, 2):
    xi = x_nodes2ii[i:i + 3]
    yi = f_nodes2ii[i:i + 3]
    coeffs = np.polyfit(xi, yi, 2)
    poly = np.poly1d(coeffs)
    x_parab = np.linspace(xi[0], xi[-1], 100)
    plt.plot(x_parab, poly(x_parab), 'g--', linewidth=2, label='Аппроксимация параболой' if i == 0 else "")
    plt.fill_between(x_parab, poly(x_parab), color='green', alpha=0.2)

plt.title("Правило Симпсона 1/3 для $\sqrt{\\cos(\\theta)}$")
plt.xlabel("$\\theta$")
plt.ylabel("$\sqrt{\\cos(\\theta)}$")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------
# 3. Правило Симпсона 3/8

## (i) для ∫₀^9 1/(1+x³) dx
def f3(x):
    return 1 / (1 + x ** 3)


a3, b3 = 0.0, 9.0
n3 = 9  # число интервалов должно быть кратно 3
h3 = (b3 - a3) / n3
x_nodes3 = np.linspace(a3, b3, n3 + 1)
f_nodes3 = f3(x_nodes3)

simpson3_8_i = 0
for i in range(0, n3, 3):
    simpson3_8_i += (3 * h3 / 8) * (f_nodes3[i] + 3 * f_nodes3[i + 1] + 3 * f_nodes3[i + 2] + f_nodes3[i + 3])
print("\nProblem 3(i). ∫₀^9 1/(1+x³) dx by Simpson’s 3/8 rule =", simpson3_8_i)

# Визуализация: для каждого набора из 4 узлов (3 отрезка) аппроксимируем кубиком
plt.figure(figsize=(8, 5))
x_dense3 = np.linspace(a3, b3, 400)
y_dense3 = f3(x_dense3)
plt.plot(x_dense3, y_dense3, 'b-', label='$1/(1+x^3)$')
plt.plot(x_nodes3, f_nodes3, 'ro', label='Узлы')

for i in range(0, n3, 3):
    xi = x_nodes3[i:i + 4]
    yi = f_nodes3[i:i + 4]
    coeffs = np.polyfit(xi, yi, 3)  # кубическая аппроксимация
    poly3 = np.poly1d(coeffs)
    x_cubic = np.linspace(xi[0], xi[-1], 100)
    plt.plot(x_cubic, poly3(x_cubic), 'm--', linewidth=2, label='Кубическая аппроксимация' if i == 0 else "")
    plt.fill_between(x_cubic, poly3(x_cubic), color='magenta', alpha=0.2)

plt.title("Правило Симпсона 3/8 для $1/(1+x^3)$")
plt.xlabel("x")
plt.ylabel("$1/(1+x^3)$")
plt.legend()
plt.grid(True)
plt.show()

## (ii) для ∫₀^(π/2) sin x dx
a3ii, b3ii = 0.0, math.pi / 2
n3ii = 3  # n должно быть кратно 3
h3ii = (b3ii - a3ii) / n3ii
x_nodes3ii = np.linspace(a3ii, b3ii, n3ii + 1)
f_nodes3ii = np.sin(x_nodes3ii)

simpson3_8_ii = (3 * h3ii / 8) * (f_nodes3ii[0] + 3 * f_nodes3ii[1] + 3 * f_nodes3ii[2] + f_nodes3ii[3])
print("Problem 3(ii). ∫₀^(π/2) sin x dx by Simpson’s 3/8 rule =", simpson3_8_ii)
print("Exact value =", 1.0)

plt.figure(figsize=(8, 5))
x_dense3ii = np.linspace(a3ii, b3ii, 400)
y_dense3ii = np.sin(x_dense3ii)
plt.plot(x_dense3ii, y_dense3ii, 'b-', label='$\sin(x)$')
plt.plot(x_nodes3ii, f_nodes3ii, 'ro', label='Узлы')

xi = x_nodes3ii
yi = f_nodes3ii
coeffs = np.polyfit(xi, yi, 3)  # кубическая аппроксимация
poly3 = np.poly1d(coeffs)
x_cubic = np.linspace(xi[0], xi[-1], 100)
plt.plot(x_cubic, poly3(x_cubic), 'm--', linewidth=2, label='Кубическая аппроксимация')
plt.fill_between(x_cubic, poly3(x_cubic), color='magenta', alpha=0.2)

plt.title("Правило Симпсона 3/8 для $\sin(x)$")
plt.xlabel("x")
plt.ylabel("$\sin(x)$")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------
# 4. Сравнение методов для ∫₀¹ 1/(1+x) dx

def f4(x):
    return 1 / (1 + x)


a4, b4 = 0.0, 1.0

# (i) Трапеции
n4 = 4
h4 = (b4 - a4) / n4
x_nodes4 = np.linspace(a4, b4, n4 + 1)
f_nodes4 = f4(x_nodes4)
trapz4 = (h4 / 2) * (f_nodes4[0] + 2 * np.sum(f_nodes4[1:-1]) + f_nodes4[-1])
print("\nProblem 4(i). ∫₀¹ 1/(1+x) dx by the Trapezoidal rule =", trapz4)

# (ii) Simpson 1/3
simpson1_3_4 = (h4 / 3) * (f_nodes4[0] + f_nodes4[-1] +
                           4 * np.sum(f_nodes4[1:-1:2]) +
                           2 * np.sum(f_nodes4[2:-1:2]))
print("Problem 4(ii). ∫₀¹ 1/(1+x) dx by Simpson’s 1/3 rule =", simpson1_3_4)

# (iii) Simpson 3/8 (n=3)
n4_38 = 3
h4_38 = (b4 - a4) / n4_38
x_nodes4_38 = np.linspace(a4, b4, n4_38 + 1)
f_nodes4_38 = f4(x_nodes4_38)
simpson3_8_4 = (3 * h4_38 / 8) * (f_nodes4_38[0] + 3 * f_nodes4_38[1] + 3 * f_nodes4_38[2] + f_nodes4_38[3])
print("Problem 4(iii). ∫₀¹ 1/(1+x) dx by Simpson’s 3/8 rule =", simpson3_8_4)

exact_val = np.log(2)
print("Exact value of ∫₀¹ 1/(1+x) dx =", exact_val)

# Визуализация: функция f4(x) и узлы
plt.figure(figsize=(8, 5))
x_dense4 = np.linspace(a4, b4, 400)
y_dense4 = f4(x_dense4)
plt.plot(x_dense4, y_dense4, 'b-', label='$1/(1+x)$')
plt.plot(x_nodes4, f_nodes4, 'ro', label='Узлы (для трапеций и Simpson 1/3)')
plt.title("Интегрирование $1/(1+x)$: сравнение методов")
plt.xlabel("x")
plt.ylabel("$1/(1+x)$")
plt.legend()
plt.grid(True)
plt.show()
