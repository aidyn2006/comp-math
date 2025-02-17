import math

def is_square_matrix(matrix):
    if len(matrix) == 0:
        return False
    is_square = True
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            is_square = False
    return is_square

def get_minor(matrix, row, col):
    minor = []
    for i in range(len(matrix)):
        if i != row:
            minor_row = []
            for j in range(len(matrix[i])):
                if j != col:
                    minor_row.append(matrix[i][j])
            minor.append(minor_row)
    return minor

def find_determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = 0
    for col in range(len(matrix[0])):
        minor = get_minor(matrix, 0, col)
        cofactor = ((-1) ** col) * matrix[0][col] * find_determinant(minor)
        determinant += cofactor
    return determinant

def transpose_matrix(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transposed.append(row)
    return transposed

'''
Inverse matrix = adjugate matrix / det(A)
adjugate matrix = transposed(cofactor matrix)
cofactor(col, row) = det(minor(matrix, col, row) * (-1) ** (row + column)
'''
def find_inverse_matrix(matrix):
    if not is_square_matrix(matrix):
        return None
    determinant = find_determinant(matrix)
    if determinant == 0:
        return None
    cofactor_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            sign = (-1) ** (i + j)
            minor = get_minor(matrix, i, j)
            determinant = find_determinant(minor)
            row.append(sign * determinant)
        cofactor_matrix.append(row)

    adjugate_matrix = transpose_matrix(cofactor_matrix)

    inverse = []

    for i in range(len(adjugate_matrix)):
        row = []
        for j in range(len(adjugate_matrix[0])):
            value = adjugate_matrix[i][j] / determinant
            row.append(value)
        inverse.append(row)

    return inverse

def solve_cubic(a, b, c, d):
    b, c, d = b / a, c / a, d / a
    p = c - (b**2) / 3
    q = (2 * b**3) / 27 - (b * c) / 3 + d
    discriminant = (q**2) / 4 + (p**3) / 27
    roots = []
    if discriminant > 0:
        sqrt_disc = math.sqrt(discriminant)
        u = (-q / 2 + sqrt_disc)**(1 / 3)
        v = (-q / 2 - sqrt_disc)**(1 / 3)
        roots.append(u + v - b / 3)
    elif discriminant == 0:
        u = (-q / 2)**(1 / 3)
        roots.extend([2 * u - b / 3, -u - b / 3])
    else:
        r = math.sqrt(-(p**3) / 27)
        phi = math.acos(-q / (2 * r))
        r = (-p / 3)**0.5
        roots.extend([2 * r * math.cos(phi / 3) - b / 3,
                      2 * r * math.cos((phi / 3) + (2 * math.pi / 3)) - b / 3,
                      2 * r * math.cos((phi / 3) + (4 * math.pi / 3)) - b / 3])
    return roots

'''
Ax = λx
λ - eigenvalue, x - eigenvector
det(A−λI)=0



'''
def find_eigenvalues_2x2(matrix):
    if len(matrix) != 2 or not is_square_matrix(matrix):
        print("The matrix is not in correct format")
        return None
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = math.sqrt(trace**2 - 4 * determinant)
    return [0.5 * (trace + discriminant), 0.5 * (trace - discriminant)]

def find_eigenvalues_3x3(matrix):
    if len(matrix) != 3 or not is_square_matrix(matrix):
        print("The matrix is not in correct format")
        return None
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    trace = a + e + i
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    sum_of_minors = (a * e - b * d) + (a * i - c * g) + (e * i - f * h)
    coeff_a = -1
    coeff_b = trace
    coeff_c = -sum_of_minors
    coeff_d = determinant
    return solve_cubic(coeff_a, coeff_b, coeff_c, coeff_d)


def cramer(coeff_matrix, constants):
    det_main = find_determinant(coeff_matrix)
    if math.isclose(det_main, 0):
        return "No unique solution (determinant is zero)"

    solution = []
    for col in range(len(constants)):
        temp_matrix = [row[:] for row in coeff_matrix]
        for row in range(len(temp_matrix)):
            temp_matrix[row][col] = constants[row]
        solution.append(find_determinant(temp_matrix) / det_main)
    return solution



def find_eigenvector(matrix, eigenvalue):
    if len(matrix) == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        if b != 0:
            x = 1
            y = (eigenvalue - a) / b
        else:
            x = (eigenvalue - d) / c
            y = 1
        return [x, y]
    elif len(matrix) == 3:
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        coeff_matrix = [
            [a - eigenvalue, b, c],
            [d, e - eigenvalue, f],
            [g, h, i - eigenvalue]
        ]
        constants = [0, 0, 0]
        eigenvector = cramer(coeff_matrix, constants)
        return eigenvector


matrix_2x2 = [
    [4, 2],
    [1, 3]
]
matrix_3x3 = [
    [6, 2, 1],
    [2, 3, 1],
    [1, 8, 1]
]

if __name__ == "__main__":
    eigenvalues_2x2 = find_eigenvalues_2x2(matrix_2x2)
    eigenvalues_3x3 = find_eigenvalues_3x3(matrix_3x3)

    print("2x2 Matrix Eigenvalues:", eigenvalues_2x2)
    print("3x3 Matrix Eigenvalues:", eigenvalues_3x3)

    for eigenvalue in eigenvalues_2x2:
        eigenvector = find_eigenvector(matrix_2x2, eigenvalue)
        print(f"2x2 Matrix Eigenvector for eigenvalue {eigenvalue}:", eigenvector)

    for eigenvalue in eigenvalues_3x3:
        eigenvector = find_eigenvector(matrix_3x3, eigenvalue)
        print(f"3x3 Matrix Eigenvector for eigenvalue {eigenvalue}:", eigenvector)

    inverse_2x2 = find_inverse_matrix(matrix_2x2)
    inverse_3x3 = find_inverse_matrix(matrix_3x3)


    print("\nInverse of 2x2 Matrix:")
    for rw in inverse_2x2:
        print(rw)
    print("Inverse of 3x3 Matrix:")
    for rw in inverse_3x3:
        print(rw)


'''

matrix = []
n = int(input("Enter the number of rows: "))
for i in range(n):
    row = input("Enter the row: ")
    row = list(map(int,row.strip().split(" ")))
    matrix.append(row)

inverse = find_inverse_matrix(matrix)
print("Inverse of Matrix:")
for i in inverse:
    print(i)
if n == 2:
    print("2x2 Matrix Eigenvalues:", find_eigenvalues_2x2(matrix))
elif n == 3:
    print("3x3 Matrix Eigenvalues:", find_eigenvalues_3x3(matrix))

'''