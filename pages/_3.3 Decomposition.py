import streamlit as st
import numpy as np
import pandas as pd


def upper_triangular(M):
    M = M.copy()

    elimination = {}
    for i in range(0, M.shape[0] - 1):
        pivot = M[i][i]

        # if pivot == 0:
        #     return M

        for j in range(i + 1, M.shape[0]):
            elimination[f'{i}{j}'] = M[j][i] / M[i][i]
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])
    return M, elimination


def lower_triangular(M, elimination):
    M = M.copy()
    for i in range(0, M.shape[0]):
        for j in range(0, M.shape[1]):
            if j > i:
                M[i][j] = 0
            elif i == j:
                M[i][j] = 1
            else:
                M[i][j] = float(elimination[f'{j}{i}'])
    return M


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


st.title('Decomposition')

variable_num = int(st.text_input("number of variables", value=3))

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

if st.button('Go'):
    augmented_matrix = edited_df.values
    M = augmented_matrix[:, :-1]
    b = augmented_matrix[:, -1].reshape(-1, 1)

    st.header('Decomposition phase')

    U, elimination = upper_triangular(M)
    st.text('U:')
    matrix_to_latex(U)

    L = lower_triangular(M, elimination)
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