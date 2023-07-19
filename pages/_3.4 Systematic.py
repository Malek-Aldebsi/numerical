import streamlit as st
import numpy as np
import pandas as pd


def U_and_L_by_systematic(M):
    U_1_1 = M[0][0]
    U_1_2 = M[0][1]
    U_1_3 = M[0][2]
    U_1_4 = M[0][3]

    L_2_1 = M[1][0] / U_1_1
    U_2_2 = M[1][1] - U_1_2 * L_2_1
    U_2_3 = M[1][2] - U_1_3 * L_2_1
    U_2_4 = M[1][3] - U_1_4 * L_2_1

    L_3_1 = M[2][0] / U_1_1
    L_3_2 = (M[2][1] - U_1_2 * L_3_1) / U_2_2
    U_3_3 = M[2][2] - U_1_3 * L_3_1 - U_2_3 * L_3_2
    U_3_4 = M[2][3] - U_1_4 * L_3_1 - U_2_4 * L_3_2

    L_4_1 = M[3][0] / U_1_1
    L_4_2 = (M[3][1] - U_1_2 * L_4_1) / U_2_2
    L_4_3 = (M[3][2] - U_1_3 * L_4_1 - U_2_3 * L_4_2) / U_3_3
    U_4_4 = M[3][3] - U_1_4 * L_4_1 - U_2_4 * L_4_2 - U_3_4 * L_4_3

    U = np.array([
        [U_1_1, U_1_2, U_1_3, U_1_4],
        [0, U_2_2, U_2_3, U_2_4],
        [0, 0, U_3_3, U_3_4],
        [0, 0, 0, U_4_4]
    ])
    L = np.array([
        [1, 0, 0, 0],
        [L_2_1, 1, 0, 0],
        [L_3_1, L_3_2, 1, 0],
        [L_4_1, L_4_2, L_4_3, 1]
    ])
    return U, L


def forward_substitution(b, L):
    n = L.shape[0]
    d = np.zeros(n)

    for i in range(n):
        d[i] = (b[i] - np.dot(L[i, :i], d[:i])) / L[i, i]
    return d


def backward_substitution(U, D):
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (D[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


def matrix_to_latex(matrix):
    temp = ''
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if col != matrix.shape[1] - 1:
                temp += f'{matrix[row][col]} &'
            else:
                temp += f'{matrix[row][col]}'
        if row != matrix.shape[0] - 1:
            temp += r'\\'

    latex = r'\begin{bmatrix}' + temp + r'\end{bmatrix}'
    st.latex(latex)


st.title('Systematic')

variable_num = 4

headers = []
values = []
equ_names = []
for i in range(variable_num):
    headers.append(f'x{i + 1}')
    equ_names.append(f'equation #{i + 1}')
    values.append(0.0)

headers.append(f'b')
data = {}
for i in headers:
    data[i] = values

df = pd.DataFrame(data)

row_names = equ_names
df['equations'] = row_names
df.set_index('equations', inplace=True)

edited_df = st.data_editor(df)

if st.button('go'):
    augmented_matrix = edited_df.values
    M = augmented_matrix[:, :-1]
    b = augmented_matrix[:, -1].reshape(-1, 1)

    st.header('Decomposition phase')

    U, L = U_and_L_by_systematic(M)
    st.text('U:')
    matrix_to_latex(U)

    st.text('L:')
    matrix_to_latex(L)

    st.header('Substitution phase')
    D = forward_substitution(b, L).reshape(-1, 1)
    st.text('D:')
    matrix_to_latex(D)

    x = backward_substitution(U, D)

    sol_txt = fr'x1 = {x[0]}'
    for i in range(1, variable_num):
        sol_txt += f', x{i + 1} = {x[i]}'
    st.text('solutions: ' + sol_txt)
