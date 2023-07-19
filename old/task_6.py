import sympy as sp
import numpy as np


# form augmented matrix
def matrix_representation(system, syms):
    # extract equation coefficients and constant
    a, b = sp.linear_eq_to_matrix(system, syms)

    # insert right hand size values into coefficients matrix
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


def upper_triangular_without_scaling_or_pivoting(M):
    # iterate over matrix rows
    for i in range(0, M.shape[0] - 1):
        # select pivot value
        pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * float(M[j][i] / M[i][i])
    # return upper triangular matrix
    return M


def upper_triangular_with_simple_pivoting(M):
    # iterate over matrix rows
    for i in range(0, M.shape[0] - 1):
        # initialize row-swap iterator
        j = 1

        # select pivot value
        pivot = M[i][i]

        # find next non-zero leading coefficient
        while pivot == 0 and i + j < M.shape[0]:
            # perform row swap operation
            if M[i + j][i] != 0:
                print(M, '\n')
                M[[i, i + j]] = M[[i + j, i]]
                print(M, '\n')
            # increment row-swap iterator
            j += 1

            # get new pivot
            pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])

    # return upper triangular matrix
    return M


def upper_triangular_with_partial_pivoting(M):
    # iterate over matrix rows
    for i in range(0, M.shape[0] - 1):
        # find largest pivot
        max_pivot_index = i
        for j in range(i + 1, M.shape[0]):
            if M[j][i] > M[max_pivot_index][i]:
                max_pivot_index = j

        print(M, '\n')
        # perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        pivot = M[i][i]
        print(M, '\n')
        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])

    # return upper triangular matrix
    return M


def upper_triangular_with_online_approach(M):
    # iterate over matrix rows
    for i in range(0, M.shape[0] - 1):
        # perform scaling
        for row in range(i, M.shape[0]):
            max_scaled_factor = abs(M[row][i])
            for column in range(i, M.shape[1] - 1):
                if abs(M[row][column]) > max_scaled_factor:
                    max_scaled_factor = abs(M[row][column])
            print(M, '\n')
            M[row] /= max_scaled_factor
            print(M, '\n')
        # find largest pivot
        max_pivot_index = i
        for j in range(i + 1, M.shape[0]):
            if M[j][i] > M[max_pivot_index][i]:
                max_pivot_index = j

        # perform row swap operation
        print(M, '\n')
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        print(M, '\n')
        pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])

    # return upper triangular matrix
    return M


def upper_triangular_with_offline_approach(M):
    # iterate over matrix rows
    for i in range(0, M.shape[0] - 1):
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
        pivot = M[i][i]
        print(M, '\n')
        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])

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


# def validate(M):
#     for i in range(0, M.shape[0] - 1):
#         if M[i][i] == 0:
#             return False
#     return True
#
#
# def validate_solution(system, solutions, tolerance=1e-6):
#     # iterate over each equation
#     for eqn in system:
#         # assert equation is solved
#         assert eqn.subs(solutions) < tolerance


# symbolic variables
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
symbolic_vars = [x1, x2, x3, x4]

# x1, x2, x3 = sp.symbols('x1 x2 x3')
# symbolic_vars = [x1, x2, x3]

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

print('\nupper triangular matrix without scaling or pivoting:')
upper_triangular_matrix = upper_triangular_without_scaling_or_pivoting(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

print('\nupper triangular matrix with simple pivoting:')
upper_triangular_matrix = upper_triangular_with_simple_pivoting(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

print('\nupper triangular matrix with partial pivoting:')
upper_triangular_matrix = upper_triangular_with_partial_pivoting(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

print('\nupper triangular matrix with online approach:')
upper_triangular_matrix = upper_triangular_with_online_approach(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

print('\nupper triangular matrix with offline approach:')
upper_triangular_matrix = upper_triangular_with_offline_approach(augmented_matrix.copy())
print(upper_triangular_matrix)
# print('solutions: ', back_substitution(upper_triangular_matrix, symbolic_vars))

# # remove zero rows
# back_sub_matrix = upper_triangular_matrix[np.any(upper_triangular_matrix != 0, axis=1)]
#
# # initialise numerical solution
# numeric_solution = np.array([0., 0., 0.])
#
# # assert that number of rows in matrix equals number of unknown variables
# if back_sub_matrix.shape[0] != len(symbolic_vars):
#     print('dependent system. infinite number of solutions')
# elif not np.any(back_sub_matrix[-1][:len(symbolic_vars)]):
#     print('inconsistent system. no solution..')
# else:
#     print(back_sub_matrix)
#     if validate(upper_triangular_matrix):
#         # back substitution to solve for variables
#         numeric_solution = back_substitution(upper_triangular_matrix, symbolic_vars)
#         print(f'\nsolutions:\n{numeric_solution}')
