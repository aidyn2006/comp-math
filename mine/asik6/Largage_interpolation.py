def lagrange_interpolation(x, y, value):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (value - x[j]) / (x[i] - x[j])
        result += term
    return result

x_values = [2, 3, 4, 6]
y_values = [45.0, 49.2, 54.1, 67.4]

missing_value = lagrange_interpolation(x_values, y_values, 5)
print(f"Пропущенное значение: {missing_value:.2f}")