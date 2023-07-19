import sympy
import numpy as np


def upper_triangular(M):
    M = M.copy()

    elimination = {}
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
            elimination[f'{i}{j}'] = M[j][i] / M[i][i]
            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])
            print(M, '\n')
    # return upper triangular matrix
    return M, elimination


def lower_triangular(M, elimination):
    M = M.copy()
    for i in range(0, M.shape[0]):
        for j in range(0, M.shape[1]):
            print(M, '\n')
            if j > i:
                M[i][j] = 0
            elif i == j:
                M[i][j] = 1
            else:
                M[i][j] = float(elimination[f'{j}{i}'])
    return M


def U_and_L_by_systematic(M):

    U_1_1 = M[0][0]
    print(f'U_1_1 = {U_1_1}')
    U_1_2 = M[0][1]
    print(f'U_1_2 = {U_1_2}')
    U_1_3 = M[0][2]
    print(f'U_1_3 = {U_1_3}')
    U_1_4 = M[0][3]
    print(f'U_1_4 = {U_1_4}')

    L_2_1 = M[1][0] / U_1_1
    print(f'L_2_1 = {L_2_1}')
    U_2_2 = M[1][1] - U_1_2 * L_2_1
    print(f'U_2_2 = {U_2_2}')
    U_2_3 = M[1][2] - U_1_3 * L_2_1
    print(f'U_2_3 = {U_2_3}')
    U_2_4 = M[1][3] - U_1_4 * L_2_1
    print(f'U_2_4 = {U_2_4}')

    L_3_1 = M[2][0] / U_1_1
    print(f'L_3_1 = {L_3_1}')
    L_3_2 = (M[2][1] - U_1_2 * L_3_1) / U_2_2
    print(f'L_3_2 = {L_3_2}')
    U_3_3 = M[2][2] - U_1_3 * L_3_1 - U_2_3 * L_3_2
    print(f'U_3_3 = {U_3_3}')
    U_3_4 = M[2][3] - U_1_4 * L_3_1 - U_2_4 * L_3_2
    print(f'U_3_4 = {U_3_4}')

    L_4_1 = M[3][0] / U_1_1
    print(f'L_4_1 = {L_4_1}')
    L_4_2 = (M[3][1] - U_1_2 * L_4_1) / U_2_2
    print(f'L_4_2 = {L_4_2}')
    L_4_3 = (M[3][2] - U_1_3 * L_4_1 - U_2_3 * L_4_2) / U_3_3
    print(f'L_4_3 = {L_4_3}')
    U_4_4 = M[3][3] - U_1_4 * L_4_1 - U_2_4 * L_4_2 - U_3_4 * L_4_3
    print(f'U_4_4 = {U_4_4}')

    U = [
        [U_1_1, U_1_2, U_1_3, U_1_4],
        [0, U_2_2, U_2_3, U_2_4],
        [0, 0, U_3_3, U_3_4],
        [0, 0, 0, U_4_4]
    ]
    L = [
        [1, 0, 0, 0],
        [L_2_1, 1, 0, 0],
        [L_3_1, L_3_2, 1, 0],
        [L_4_1, L_4_2, L_4_3, 1]
    ]
    return U, L


M = np.array([[2., -1., 2., 2.], [2., -2., -2., -2.], [2., -2., 2., 1.], [2., -3., 2., 0.]])
b = np.array([[0], [-2], [-3], [-5]])

print('\nupper triangular matrix:\n')
U, elimination = upper_triangular(M)
print(U)
print('\nlower triangular matrix:\n')
L = lower_triangular(M, elimination)
print(L)

U, L = U_and_L_by_systematic(M)
print('upper triangular matrix:\n', U)
print('lower triangular matrix:\n', L)

