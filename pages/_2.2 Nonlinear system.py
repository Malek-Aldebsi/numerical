from docxtpl import DocxTemplate
from sympy import symbols, sympify, diff, N, E, Matrix

import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from docx import Document
from docx.shared import Inches


def draw_approximate_and_true_errors(errors):
    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)(x)")
    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Et(%)(y)")
    plt.plot(errors.keys(), [error[2] for error in errors.values()], label="Ea(%)(x)")
    plt.plot(errors.keys(), [error[3] for error in errors.values()], label="Ea(%)(y)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for Newton Raphson non linear system method')

    plt.savefig('Newton Raphson For System.png')

    st.image(f'Newton Raphson For System.png', use_column_width=True)

    plt.close()


def newton_raphson_for_non_linear_system(f_1, f_2, x_exact, y_exact, x_i, y_i, approximate_relative_error_cond_x=0.005, approximate_relative_error_cond_y=0.005, true_relative_error_cond_x=0.005, true_relative_error_cond_y=0.005, iter_num_cond=10):
    x, y, e = symbols('x y e')
    f_1 = N(f_1.subs(e, E))
    f_2 = N(f_2.subs(e, E))

    X = Matrix([[x_i], [y_i]])
    F = Matrix([[N(f_1.subs({x: X[0], y: X[1]}))], [N(f_2.subs({x: X[0], y: X[1]}))]])

    df1_dx = diff(f_1, x, 1)
    df1_dy = diff(f_1, y, 1)
    df2_dx = diff(f_2, x, 1)
    df2_dy = diff(f_2, y, 1)

    J_x = Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])

    data = {
        'x': [],
        'y': [],
        'f1(x, y)': [],
        'f2(x, y)': [],
        'df1_dx': [],
        'df1_dy': [],
        'df2_dx': [],
        'df2_dy': [],
        'Et(%)(x)': [],
        'Et(%)(y)': [],
        'Ea(%)(x)': [],
        'Ea(%)(y)': [],
    }

    iter_num = 0
    true_relative_error_x = abs((x_exact - X[0]) / x_exact)
    true_relative_error_y = abs((y_exact - X[1]) / y_exact)
    approximate_relative_error_x = 10
    approximate_relative_error_y = 10

    errors = {}
    while (
            true_relative_error_x > true_relative_error_cond_x or true_relative_error_y > true_relative_error_cond_y) and (
            approximate_relative_error_x > approximate_relative_error_cond_x or approximate_relative_error_y > approximate_relative_error_cond_y) and iter_num <= iter_num_cond:
        data['x'].append(X[0])
        data['y'].append(X[1])
        data['f1(x, y)'].append(F[0])
        data['f2(x, y)'].append(F[1])
        data['Et(%)(x)'].append(true_relative_error_x)
        data['Et(%)(y)'].append(true_relative_error_y)
        data['Ea(%)(x)'].append(approximate_relative_error_x if iter_num != 0 else '---')
        data['Ea(%)(y)'].append(approximate_relative_error_x if iter_num != 0 else '---')

        df1_dx = N(J_x[0, 0].subs({x: X[0], y: X[1]}))
        df1_dy = N(J_x[0, 1].subs({x: X[0], y: X[1]}))
        df2_dx = N(J_x[1, 0].subs({x: X[0], y: X[1]}))
        df2_dy = N(J_x[1, 1].subs({x: X[0], y: X[1]}))

        data['df1_dx'].append(df1_dx)
        data['df1_dy'].append(df1_dy)
        data['df2_dx'].append(df2_dx)
        data['df2_dy'].append(df2_dy)

        J = Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])
        J_inv = J.inv()

        Y = J_inv * (-F)

        pre_x_i = X[0]
        pre_y_i = X[1]

        X = X + Y
        F = Matrix([[N(f_1.subs({x: X[0], y: X[1]}))], [N(f_2.subs({x: X[0], y: X[1]}))]])

        iter_num += 1
        true_relative_error_x = abs((x_exact - X[0]) / x_exact)
        true_relative_error_y = abs((y_exact - X[1]) / y_exact)
        approximate_relative_error_x = abs((X[0] - pre_x_i) / X[0])
        approximate_relative_error_y = abs((X[1] - pre_y_i) / X[1])

        errors[iter_num] = [true_relative_error_x, true_relative_error_y, approximate_relative_error_x,
                            approximate_relative_error_y]
    return data, errors


def get_newton_raphson_for_non_linear_system_doc(f_x, f_x_2, iteration, approximate_cond, data):
    main_path = r"templetes/newton_raphson_for_system.docx"
    template = DocxTemplate(main_path)

    x, y = symbols('x y')

    context = {
        "fun1": f_x,
        "fun2": f_x_2,
        "iteration": iteration,
        "approximate": approximate_cond,

        "method": 'Newton Raphson for non linear',

        "x0": round(data['x'][0], 5),
        "y0": round(data['y'][0], 5),

        "funx1": diff(f_1, x, 1),
        "funx2": diff(f_2, x, 1),
        "funy1": diff(f_1, y, 1),
        "funy2": diff(f_2, y, 1),

        "i1": round(data['df1_dx'][0], 5),
        "i2": round(data['df1_dy'][0], 5),
        "i3": round(data['df2_dx'][0], 5),
        "i4": round(data['df2_dy'][0], 5),

        "f1": round(data['f1(x, y)'][0], 5),
        "f2": round(data['f2(x, y)'][0], 5),

        "Y1": round(data['y'][1], 5),
        "Y2": round(data['y'][2], 5),

        "X1": round(data['x'][1], 5),
        "X2": round(data['x'][2], 5),

        "i5": round(data['df1_dx'][1], 5),
        "i6": round(data['df1_dy'][1], 5),
        "i7": round(data['df2_dx'][1], 5),
        "i8": round(data['df2_dy'][1], 5),

        "f3": round(data['f1(x, y)'][1], 5),
        "f4": round(data['f2(x, y)'][1], 5),

        "Y3": round(data['y'][3], 5),
        "Y4": round(data['y'][4], 5),

        "X3": round(data['x'][3], 5),
        "X4": round(data['x'][4], 5),

        "ea1": data['Ea(%)(x)'][0],

        "i9": round(data['df1_dx'][2], 5),
        "i10": round(data['df1_dy'][2], 5),
        "i11": round(data['df2_dx'][2], 5),
        "i12": round(data['df2_dy'][2], 5),

        "f5": round(data['f1(x, y)'][2], 5),
        "f6": round(data['f2(x, y)'][2], 5),

        "Y5": round(data['y'][5], 5),
        "Y6": round(data['y'][6], 5),

        "ea2": round(data['Ea(%)(x)'][1], 5),

        "xreal": round(data['x'][-1], 5),
        "yreal": round(data['y'][-1], 5),
    }

    template.render(context)

    headers = ['iteration'] + list(data.keys())
    table = template.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'

    header_cells = table.rows[0].cells
    for index, header in enumerate(headers):
        header_cells[index].text = header

    for row in range(len(list(data.values())[0])):
        row_cells = table.add_row().cells
        for index, column in enumerate(headers):
            row_cells[index].text = str(data[column][row] if column != 'iteration' else row)

    template.add_picture('Newton Raphson For System.png', width=Inches(5))

    template.save('Newton Raphson For System.docx')


st.title('Newton Raphson Method for non linear system')

with st.form("Newton_Raphson_fpr_non_linear_system_form"):
    col1, col2 = st.columns(2)
    with col1:
        f_1 = st.text_input("first function", value=r"x**2 + y - x - 0.4")
        f_1 = sympify(f_1)
        st.latex(f_1)

    with col2:
        f_2 = st.text_input("second function", value=r"2 * y + 3 * x * y - 2 * x**3")
        f_2 = sympify(f_2)
        st.latex(f_2)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x_i = st.text_input("initial value for x", value=-5)
        x_i = float(x_i)

        approximate_relative_error_cond_x = st.text_input("approximate error for x", value=0.005)
        approximate_relative_error_cond_x = float(approximate_relative_error_cond_x)

        iter_num_cond = st.text_input("iteration number", value=10)
        iter_num_cond = int(iter_num_cond)
    with col2:
        x_exact = st.text_input("exact solution for x", value=-0.55477)
        x_exact = float(x_exact)

        true_relative_error_cond_x = st.text_input("true error for x", value=0.005)
        true_relative_error_cond_x = float(true_relative_error_cond_x)

    with col3:
        y_i = st.text_input("initial value for y", value=0)
        y_i = float(y_i)

        approximate_relative_error_cond_y = st.text_input("approximate error for y", value=0.005)
        approximate_relative_error_cond_y = float(approximate_relative_error_cond_y)
    with col4:
        y_exact = st.text_input("exact solution for y", value=-1.0173242)
        y_exact = float(y_exact)

        true_relative_error_cond_y = st.text_input("true error for y", value=0.005)
        true_relative_error_cond_y = float(true_relative_error_cond_y)

    submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        data, errors = newton_raphson_for_non_linear_system(f_1, f_2, x_exact, y_exact, x_i, y_i, approximate_relative_error_cond_x, approximate_relative_error_cond_y, true_relative_error_cond_x, true_relative_error_cond_y, iter_num_cond)

        df = pd.DataFrame(data)
        df
        draw_approximate_and_true_errors(errors)

        get_newton_raphson_for_non_linear_system_doc(f_1, f_2, iter_num_cond, approximate_relative_error_cond_x, data)

st.download_button("Download Document", data=open('documentation.docx', "rb"), file_name="document.docx")
