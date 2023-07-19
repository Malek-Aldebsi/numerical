import sympy as sp
import pandas as pd


import streamlit as st
import numpy as np
from sympy import sympify


def matrix_representation(system, syms):
    a, b = sp.linear_eq_to_matrix(system, syms)
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)


def upper_triangular_without_scaling_or_pivoting(M):
    for i in range(0, M.shape[0] - 1):
        pivot = M[i][i]

        if pivot == 0:
            return M

        for j in range(i + 1, M.shape[0]):
            # print(M, '\n')
            # print(float(M[j][i] / M[i][i]))
            # print(str(j + 1) + "E" + "-" + "(" + str(M[j][i]) + "/" + str(M[i][i]) + ")" + str(i + 1) + "E")
            # print(np.around(M, 5), '\n')
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
            if (i == i + j):
                print("NO SWAP")
            else:
                print("swap between " + str(i) + "E" + " and " + str(i + j) + "E")
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
        # print(float(M[j][i] / M[i][i]))
        # print(str(j+1)+"E"+"-"+"("+str(M[j][i])+"/"+str(M[i][i])+")"+str(i+1)+"E")
        # print(np.around(M,5), '\n')
        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(M, '\n')
            print(float(M[j][i] / M[i][i]))
            print(str(j + 1) + "E" + "-" + "(" + str(M[j][i]) + "/" + str(M[i][i]) + ")" + str(i + 1) + "E")
            print(np.around(M, 5), '\n')

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
        if (i == max_pivot_index):
            print("NO SWAP")
        else:
            print("swap between " + str(i) + "E" + " and " + str(max_pivot_index) + "E")
        pivot = M[i][i]
        print(M, '\n')
        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            # print(M, '\n')
            print(float(M[j][i] / M[i][i]))
            print(str(j + 1) + "E" + "-" + "(" + str(M[j][i]) + "/" + str(M[i][i]) + ")" + str(i + 1) + "E")
            print(np.around(M, 5), '\n')

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

            # print(M, '\n')
            print(str(row + 1) + "E" + " / " + str(max_scaled_factor))
            M[row] /= max_scaled_factor
            print(M, '\n')
        # find largest pivot
        max_pivot_index = i
        for j in range(i + 1, M.shape[0]):
            if M[j][i] > M[max_pivot_index][i]:
                max_pivot_index = j

        # perform row swap operation
        # print(M, '\n')
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        if (i == max_pivot_index):
            print("NO SWAP")
        else:
            print("swap between " + str(i) + "E" + " and " + str(max_pivot_index) + "E")
        print(M, '\n')
        pivot = M[i][i]

        # if pivot is zero, remaining rows are all zeros
        if pivot == 0:
            # return upper triangular matrix
            return M

        # iterate over remaining rows
        for j in range(i + 1, M.shape[0]):
            print(float(M[j][i] / M[i][i]))
            print(str(j + 1) + "E" + "-" + "(" + str(M[j][i]) + "/" + str(M[i][i]) + ")" + str(i + 1) + "E")
            # print(np.around(M,5), '\n')

            # subtract current row from remaining rows
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])
            print(M, '\n')
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
            print(str(row + 1) + "E" + " / " + str(max_scaled_factor))
            temp_M[row] /= max_scaled_factor
            print(temp_M, '\n')
        # find largest pivot
        max_pivot_index = i
        for j in range(i + 1, temp_M.shape[0]):
            if temp_M[j][i] > temp_M[max_pivot_index][i]:
                max_pivot_index = j

        print(M, '\n')
        # perform row swap operation
        M[[i, max_pivot_index]] = M[[max_pivot_index, i]]
        if (i == max_pivot_index):
            print("NO SWAP")
        else:
            print("swap between " + str(i) + "E" + " and " + str(max_pivot_index) + "E")
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
            print(float(M[j][i] / M[i][i]))
            print(str(j + 1) + "E" + "-" + "(" + str(M[j][i]) + "/" + str(M[i][i]) + ")" + str(i + 1) + "E")
            M[j] = M[j] - M[i] * (M[j][i] / M[i][i])

    # return upper triangular matrix
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


st.title('Gauss Elimination')

with st.form("Fixed_Point_form"):
    col1, col2 = st.columns(2)
    with col1:
        variable_num = st.text_input("Enter the number of variables", value=3)
        variable_num = int(variable_num)
        st.write('variable number:', variable_num)

    with col2:
        st.text('')
        st.text('')
        st.form_submit_button(label="change")

    equations = {}
    for i in range(variable_num):
        equations[i] = st.text_input(f"equation #{i}", value=r"0 * x1 + 0 * x2 + 0 * x3 + 0 * x4 + 0 * x5 + 0")
        equations[i] = sympify(equations[i])
        st.latex(equations[i])

    options = ['without scaling or pivoting', 'with simple pivoting', 'with partial pivoting', 'with online approach', 'with offline approach']
    selected_option = st.selectbox('Select an option:', options)

    submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
        symbolic_vars = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

        _equations = [
            sympify(equation) for index, equation in equations.items()
        ]

        augmented_matrix = matrix_representation(system=_equations, syms=symbolic_vars[:variable_num])
        st.text('augmented matrix:')
        matrix_to_latex(augmented_matrix)

        if selected_option == 'without scaling or pivoting':
            upper_triangular_without_scaling_or_pivoting = upper_triangular_without_scaling_or_pivoting(augmented_matrix.copy())
            st.text('upper triangular without scaling or pivoting:')
            matrix_to_latex(upper_triangular_without_scaling_or_pivoting)
            st.text('solutions:')
            solutions = back_substitution(upper_triangular_without_scaling_or_pivoting)
            solutions

        elif selected_option == 'with simple pivoting':
            upper_triangular_with_simple_pivoting = upper_triangular_with_simple_pivoting(
                augmented_matrix.copy())
            st.text('upper triangular with simple pivoting:')
            matrix_to_latex(upper_triangular_with_simple_pivoting)
            st.text('solutions:')
            solutions = back_substitution(upper_triangular_with_simple_pivoting)
            solutions

        elif selected_option == 'with partial pivoting':
            upper_triangular_with_partial_pivoting = upper_triangular_with_partial_pivoting(
                augmented_matrix.copy())
            st.text('upper triangular with partial pivoting:')
            matrix_to_latex(upper_triangular_with_partial_pivoting)
            st.text('solutions:')
            solutions = back_substitution(upper_triangular_with_partial_pivoting)
            solutions

        elif selected_option == 'with online approach':
            upper_triangular_with_online_approach = upper_triangular_with_online_approach(
                augmented_matrix.copy())
            st.text('upper triangular with online approach:')
            matrix_to_latex(upper_triangular_with_online_approach)
            st.text('solutions:')
            solutions = back_substitution(upper_triangular_with_online_approach)
            solutions

        elif selected_option == 'with offline approach':
            upper_triangular_with_offline_approach = upper_triangular_with_offline_approach(
                augmented_matrix.copy())
            st.text('upper triangular with offline approach:')
            matrix_to_latex(upper_triangular_with_offline_approach)
            st.text('solutions:')
            solutions = back_substitution(upper_triangular_with_offline_approach)
            solutions

        #
        # get_doc('Fixed Point Method', f_x, g_x, x_exact, x_i, approximate_relative_error_cond,
        #         true_relative_error_cond, iter_num_cond, data)
