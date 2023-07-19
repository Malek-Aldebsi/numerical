import sympy as sp
import numpy as np


# form augmented matrix
def matrix_representation(system, syms):
    # extract equation coefficients and constant
    a, b = sp.linear_eq_to_matrix(system, syms)

    # insert right hand size values into coefficients matrix
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


def upper_triangular_without_scaling_or_pivoting(M):
    print(M, '\n')
    # iterate over matrix rows
    for i in range(M.shape[0]):
        # select pivot value
        pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(M.shape[0]):
            if j != i:
                print(M, '\n')
                # subtract current row from remaining rows
                M[j] = M[j] - M[i] * (M[j][i] / M[i][i])
                print(M, '\n')
    # return upper triangular matrix
    return M


def upper_triangular_with_offline_approach(M):
    # iterate over matrix rows
    for i in range(M.shape[0]):
        temp_M = M.copy()
        # perform scaling
        for row in range(i, temp_M.shape[0]):
            max_scaled_factor = abs(temp_M[row][i])
            for column in range(i, temp_M.shape[1] - 1):
                if abs(temp_M[row][column]) > max_scaled_factor:
                    max_scaled_factor = abs(temp_M[row][column])
            temp_M[row] /= max_scaled_factor

        # find largest pivot
        max_pivot_index = i
        for j in range(i + 1, temp_M.shape[0]):
            if temp_M[j][i] > temp_M[max_pivot_index][i]:
                max_pivot_index = j

        print(M, '\n')
        # perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        print(M, '\n')
        pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(0, M.shape[0]):
            if j != i:
                print(M, '\n')
                # subtract current row from remaining rows
                M[j] = M[j] - M[i] * (M[j][i] / M[i][i])
                print(M, '\n')

    # return upper triangular matrix
    return M


def back_substitution(M, syms):
    M = M.copy()
    syms = syms.copy()
    # symbolic variable index
    for i, row in reversed(list(enumerate(M))):
        # create symbolic equation
        eqn = -M[i][-1]
        for j in range(len(syms)):
            eqn += syms[j] * row[j]

        # solve symbolic expression and store variable
        syms[i] = sp.solve(eqn, syms[i])[0]

    # return list of evaluated variables
    return syms


x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
symbolic_vars = [x1, x2, x3, x4]

# define system of equations
equations = [
    2 * x1 - x2 + 2 * x3 + 2 * x4 - 0,
    2 * x1 - 2 * x2 - 2 * x3 - 2 * x4 - -2,
    2 * x1 - 2 * x2 + 2 * x3 + x4 - -3,
    2 * x1 - 3 * x2 + 2 * x3 - -5
]

# display equations
[print(eqn) for eqn in equations]

# obtain augmented matrix representation
augmented_matrix = matrix_representation(system=equations, syms=symbolic_vars)
print('\naugmented matrix:\n', augmented_matrix)

print('\nwithout scaling or pivoting:\n')
upper_triangular_matrix = upper_triangular_without_scaling_or_pivoting(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

print('\nupper triangular matrix with offline approach:\n')
upper_triangular_matrix = upper_triangular_with_offline_approach(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))
