import sympy as sp
import pandas as pd


import streamlit as st
import numpy as np
from sympy import sympify


def matrix_representation(system, syms):
    a, b = sp.linear_eq_to_matrix(system, syms)
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


def gauss_jordan_elimination_without_scaling_or_pivoting(M):
    n = M.shape[0]

    for i in range(n):
        # Find the pivot element in the current column
        pivot = M[i][i]

        # If the pivot is zero, we need to find a non-zero pivot in the same column
        if pivot == 0:
            for k in range(i + 1, n):
                if M[k][i] != 0:
                    # Swap the rows to get a non-zero pivot
                    M[[i, k]] = M[[k, i]]
                    pivot = M[i][i]
                    break

        if pivot == 0:
            # No non-zero pivot found, move to the next column
            continue

        # Scale the current row to make the pivot element 1
        M[i] = M[i] / pivot

        # Eliminate other elements in the current column
        for j in range(n):
            if i != j:
                M[j] = M[j] - M[i] * M[j][i]

    return M


def gauss_jordan_elimination_with_simple_pivoting(M):
    for i in range(0, M.shape[0]):
        pivot = M[i][i]

        if pivot == 0:
            j = 1
            while i + j < M.shape[0]:
                if M[i + j][i] != 0:
                    # Perform row swap operation
                    M[[i, i + j]] = M[[i + j, i]]
                    break
                j += 1

            # After row swapping, the new pivot will be at M[i][i]
            pivot = M[i][i]

        if pivot == 0:
            # If all elements in this column are zero, move to the next column
            continue

        # Scale the current row to make the pivot element 1
        M[i] = M[i] / pivot

        # Eliminate other elements in the current column
        for j in range(M.shape[0]):
            if i != j:
                M[j] = M[j] - M[i] * M[j][i]

    return M


def gauss_jordan_elimination_with_partial_pivoting(M):
    for i in range(0, M.shape[0]):
        # Find the row with the largest pivot
        max_pivot_index = i
        for j in range(i + 1, M.shape[0]):
            if abs(M[j][i]) > abs(M[max_pivot_index][i]):
                max_pivot_index = j

        # Perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        pivot = M[i][i]

        if pivot == 0:
            # If all elements in this column are zero, move to the next column
            continue

        # Scale the current row to make the pivot element 1
        M[i] = M[i] / pivot

        # Eliminate other elements in the current column
        for j in range(M.shape[0]):
            if i != j:
                M[j] = M[j] - M[i] * M[j][i]

    return M


def gauss_jordan_elimination_with_online_approach(M):
    for i in range(0, M.shape[0]):
        # Perform scaling
        for row in range(i, M.shape[0]):
            max_scaled_factor = abs(M[row][i])
            for column in range(i, M.shape[1] - 1):
                if abs(M[row][column]) > max_scaled_factor:
                    max_scaled_factor = abs(M[row][column])
            M[row] /= max_scaled_factor

        # Find the row with the largest pivot
        max_pivot_index = i
        for j in range(i + 1, M.shape[0]):
            if abs(M[j][i]) > abs(M[max_pivot_index][i]):
                max_pivot_index = j

        # Perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        pivot = M[i][i]

        if pivot == 0:
            # If all elements in this column are zero, move to the next column
            continue

        # Scale the current row to make the pivot element 1
        M[i] = M[i] / pivot

        # Eliminate other elements in the current column
        for j in range(M.shape[0]):
            if i != j:
                M[j] = M[j] - M[i] * M[j][i]

    return M


def gauss_jordan_elimination_with_offline_approach(M):
    for i in range(0, M.shape[0]):
        temp_M = M.copy()

        # Perform scaling
        for row in range(i, temp_M.shape[0]):
            max_scaled_factor = abs(temp_M[row][i])
            for column in range(i, temp_M.shape[1] - 1):
                if abs(temp_M[row][column]) > max_scaled_factor:
                    max_scaled_factor = abs(temp_M[row][column])
            temp_M[row] /= max_scaled_factor

        # Find the row with the largest pivot
        max_pivot_index = i
        for j in range(i + 1, temp_M.shape[0]):
            if abs(temp_M[j][i]) > abs(temp_M[max_pivot_index][i]):
                max_pivot_index = j

        # Perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        pivot = M[i][i]

        if pivot == 0:
            # If all elements in this column are zero, move to the next column
            continue

        # Scale the current row to make the pivot element 1
        M[i] = M[i] / pivot

        # Eliminate other elements in the current column
        for j in range(M.shape[0]):
            if i != j:
                M[j] = M[j] - M[i] * M[j][i]

    return M


def back_substitution(augmented_matrix):
    n = len(augmented_matrix)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i][-1]
        for j in range(i + 1, n):
            x[i] -= augmented_matrix[i][j] * x[j]
        x[i] /= augmented_matrix[i][i]

    return x


def matrix_to_latex(matrix):
    temp = ''
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if col != matrix.shape[1]-1:
                temp += f'{matrix[row][col]} &'
            else:
                temp += f'{matrix[row][col]}'
        if row != matrix.shape[0]-1:
            temp += r'\\'

    latex = r'\begin{bmatrix}' + temp + r'\end{bmatrix}'
    st.latex(latex)


st.title('Gauss Jordan Elimination')

variable_num = int(st.text_input("number of variables", value=3))

headers = []
values = []
equ_names = []
for i in range(variable_num):
    headers.append(f'x{i+1}')
    equ_names.append(f'equation #{i+1}')
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

augmented_matrix=edited_df.values
print(augmented_matrix)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    checkbox_without_scaling_or_pivoting = st.checkbox('without scaling or pivoting')
with col2:
    checkbox_with_simple_pivoting = st.checkbox('with simple pivoting')
with col3:
    checkbox_with_partial_pivoting = st.checkbox('with partial pivoting')
with col4:
    checkbox_with_online_approach = st.checkbox('with online approach')
with col5:
    checkbox_with_offline_approach = st.checkbox('with offline approach')

if st.button('Go', use_container_width=True):
    st.header('augmented matrix:')
    matrix_to_latex(augmented_matrix)

    if checkbox_without_scaling_or_pivoting:
        st.header('Without scaling or pivoting')
        upper_triangular_without_scaling_or_pivoting = gauss_jordan_elimination_without_scaling_or_pivoting(augmented_matrix.copy())
        st.text('upper triangular without scaling or pivoting:')
        matrix_to_latex(upper_triangular_without_scaling_or_pivoting)

        solutions = back_substitution(upper_triangular_without_scaling_or_pivoting)
        sol_txt = fr'x1 = {solutions[0]}'
        for i in range(1, variable_num):
                sol_txt += f', x{i+1} = {solutions[i]}'
        st.text('solutions: ' + sol_txt)

    if checkbox_with_simple_pivoting:
        st.header('With simple pivoting')
        upper_triangular_with_simple_pivoting = gauss_jordan_elimination_with_simple_pivoting(
            augmented_matrix.copy())
        st.text('upper triangular with simple pivoting:')
        matrix_to_latex(upper_triangular_with_simple_pivoting)
        solutions = back_substitution(upper_triangular_with_simple_pivoting)
        sol_txt = fr'x1 = {solutions[0]}'
        for i in range(1, variable_num):
            sol_txt += f', x{i + 1} = {solutions[i]}'
        st.text('solutions: ' + sol_txt)

    if checkbox_with_partial_pivoting:
        st.header('With partial pivoting')
        upper_triangular_with_partial_pivoting = gauss_jordan_elimination_with_partial_pivoting(
            augmented_matrix.copy())
        st.text('upper triangular with partial pivoting:')
        matrix_to_latex(upper_triangular_with_partial_pivoting)
        solutions = back_substitution(upper_triangular_with_partial_pivoting)
        sol_txt = fr'x1 = {solutions[0]}'
        for i in range(1, variable_num):
            sol_txt += f', x{i + 1} = {solutions[i]}'
        st.text('solutions: ' + sol_txt)

    if checkbox_with_online_approach:
        st.header('With online approach')
        upper_triangular_with_online_approach = gauss_jordan_elimination_with_online_approach(
            augmented_matrix.copy())
        st.text('upper triangular with online approach:')
        matrix_to_latex(upper_triangular_with_online_approach)

        solutions = back_substitution(upper_triangular_with_online_approach)
        sol_txt = fr'x1 = {solutions[0]}'
        for i in range(1, variable_num):
            sol_txt += f', x{i + 1} = {solutions[i]}'
        st.text('solutions: ' + sol_txt)

    if checkbox_with_offline_approach:
        st.header('With offline approach')
        upper_triangular_with_offline_approach = gauss_jordan_elimination_with_offline_approach(
            augmented_matrix.copy())
        st.text('upper triangular with offline approach:')
        matrix_to_latex(upper_triangular_with_offline_approach)

        solutions = back_substitution(upper_triangular_with_offline_approach)
        sol_txt = fr'x1 = {solutions[0]}'
        for i in range(1, variable_num):
            sol_txt += f', x{i + 1} = {solutions[i]}'
        st.text('solutions: ' + sol_txt)
