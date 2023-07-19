import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols

values = {
          'a14': 0,
          'a24': 0,
          'a34': 0,
          'a44': 0,

          's1': 0,
          's2': 0,
          's3': 0,
          's4': 0,

          "xt1": 0,
          "xt2": 0,
          "xt3": 0,
          "xt4": 0,

          "x1": 0,
          "x2": 0,
          "x3": 0,
          "x4": 0,
          "l": 1.2,
          }

x1, x2, x3, x4 = symbols("x1,x2,x3,x4")
symbolic_vars = [x1, x2, x3,x4]


st.title('Iterative Methods')

variable_num = int(st.text_input("number of variables (3, 4)", value=3))

headers = []
_values = []
equ_names = []
for i in range(variable_num):
    headers.append(f'x{i+1}')
    equ_names.append(f'equation #{i+1}')
    _values.append(0.0)

headers.append(f'b')
data = {}
for i in headers:
    data[i] = _values

df = pd.DataFrame(data)

row_names = equ_names
df['equations'] = row_names
df.set_index('equations', inplace=True)

edited_df = st.data_editor(df)

augmented_matrix = edited_df.values

for i in range(augmented_matrix.shape[0]):
    for j in range(augmented_matrix.shape[1] - 1):
        values[f'a{i+1}{j+1}'] = augmented_matrix[i, j]
    values[f'b{i+1}'] = augmented_matrix[i, -1]

iter_num = int(st.text_input('iteration number', value=10))

col1, col2, col3, col4 = st.columns(4)
with col1:
    values['xt1'] = float(st.text_input('x1 true value', value=0))
with col2:
    values['xt2'] = float(st.text_input('x2 true value', value=0))
with col3:
    values['xt3'] = float(st.text_input('x3 true value', value=0))
if variable_num == 4:
    with col4:
        values['xt4'] = float(st.text_input('x4', value=0))

col1, col2, col3 = st.columns(3)
with col1:
    checkbox_gaussian_seidel = st.checkbox('Gaussian Seidel')
with col2:
    checkbox_jacopi_method = st.checkbox('Jacopi Method')
with col3:
    checkbox_relaxation = st.checkbox('Relaxation')

if checkbox_relaxation:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        values['x1'] = float(st.text_input('x1 initial value', value=0))
        values['l'] = float(st.text_input('λ', value=0))

    with col2:
        values['x2'] = float(st.text_input('x2 initial value', value=0))
    with col3:
        values['x3'] = float(st.text_input('x3 initial value', value=0))
    if variable_num == 4:
        with col4:
            values['x4'] = float(st.text_input('x4 initial value', value=0))


def gaussian_seidel():
    X1, X2, X3, X4 = [], [], [], []
    A1, A2, A3, A4 = [], [], [], []
    T1, T2, T3, T4 = [], [], [], []

    fn1 = values['a12'] * x2 + values['a13'] * x3 + values['a14'] * x4
    fn2 = values['a21'] * x1 + values['a23'] * x3 + values['a24'] * x4
    fn3 = values['a31'] * x1 + values['a32'] * x2 + values['a34'] * x4
    fn4 = values['a31'] * x1 + values['a32'] * x2 + values['a33'] * x3

    b1, b2, b3, b4 = 0, 0, 0, 0
    for k in range(iter_num):
        if values['a11'] != 0:
            old1 = b1
            b1 = (values['b1'] - fn1.subs([(x2, values['s2']), (x3, values['s3']), (x4, values['s4'])])) / values['a11']
            values['s1'] = b1
            X1.append(b1)

            t1 = (abs(values['xt1'] - b1) / values['xt1']) * 100
            T1.append(t1)

            a1 = abs((b1 - old1) / b1) * 100 if k!=0 else '---'
            A1.append(a1)

        if values['a12'] != 0:
            old2 = b2
            b2 = (values['b2'] - fn2.subs([(x1, values['s1']), (x3, values['s3']), (x4, values['s4'])])) / values["a22"]
            values['s2'] = b2
            X2.append(b2)

            t2 = (abs(values['xt2'] - b2) / values['xt2']) * 100
            T2.append(t2)

            a2 = abs((b2 - old2) / b2) * 100 if k != 0 else '---'
            A2.append(a2)

        if values['a13'] != 0:
            old3 = b3
            b3 = (values['b3'] - fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4, values['s4'])])) / values["a33"]
            values['s3'] = b3
            X3.append(b3)

            t3 = (abs(values['xt3'] - b3) / values['xt3']) * 100
            T3.append(t3)

            a3 = abs((b3 - old3) / b3) * 100 if k!=0 else '---'
            A3.append(a3)

        if values['a14'] != 0:
            old4 = b4
            b4 = (values['b4'] - fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])])) / values["a44"]
            values['x4'] = b4
            X4.append(b4)

            t4 = (abs(values['xt4'] - b4) / values['xt4']) * 100
            T4.append(t4)

            a4 = abs((b4 - old4) / b4) * 100 if k != 0 else '---'
            A4.append(a4)

    return X1, X2, X3, X4, A1, A2, A3, A4, T1, T2, T3, T4


def jacopi_method():
    X1, X2, X3, X4 = [], [], [], []
    A1, A2, A3, A4 = [], [], [], []
    T1, T2, T3, T4 = [], [], [], []

    fn1 = values['a12'] * x2 + values['a13'] * x3 + values['a14'] * x4
    fn2 = values['a21'] * x1 + values['a23'] * x3 + values['a24'] * x4
    fn3 = values['a31'] * x1 + values['a32'] * x2 + values['a34'] * x4
    fn4 = values['a31'] * x1 + values['a32'] * x2 + values['a33'] * x3

    b1, b2, b3, b4 = 0, 0, 0, 0
    for k in range(iter_num):
        if values['a11'] != 0:
            old1 = b1
            b1 = (values['b1'] - fn1.subs([(x2, values['s2']), (x3, values['s3']), (x4, values['s4'])])) / values['a11']
            X1.append(b1)

            t1 = (abs(values['xt1'] - b1) / values['xt1']) * 100
            T1.append(t1)

            a1 = abs((b1 - old1) / b1) * 100 if k!=0 else '---'
            A1.append(a1)

        if values['a12'] != 0:
            old2 = b2
            b2 = (values['b2'] - fn2.subs([(x1, values['s1']), (x3, values['s3']), (x4, values['s4'])])) / values["a22"]
            X2.append(b2)

            t2 = (abs(values['xt2'] - b2) / values['xt2']) * 100
            T2.append(t2)

            a2 = abs((b2 - old2) / b2) * 100 if k != 0 else '---'
            A2.append(a2)

        if values['a13'] != 0:
            old3 = b3
            b3 = (values['b3'] - fn3.subs([(x1, values['s1']), (x2, values['s2']), (x4, values['s4'])])) / values["a33"]
            X3.append(b3)

            t3 = (abs(values['xt3'] - b3) / values['xt3']) * 100
            T3.append(t3)

            a3 = abs((b3 - old3) / b3) * 100 if k!=0 else '---'
            A3.append(a3)

        if values['a14'] != 0:
            old4 = b4
            b4 = (values['b4'] - fn4.subs([(x1, values['s1']), (x2, values['s2']), (x3, values['s3'])])) / values["a44"]
            X4.append(b4)

            t4 = (abs(values['xt4'] - b4) / values['xt4']) * 100
            T4.append(t4)

            a4 = abs((b4 - old4) / b4) * 100 if k != 0 else '---'
            A4.append(a4)

        if values['a11'] != 0:
            values['s1'] = b1
        if values['a12'] != 0:
            values['s2'] = b2
        if values['a13'] != 0:
            values['s3'] = b3
        if values['a14'] != 0:
            values['x4'] = b4

    return X1, X2, X3, X4, A1, A2, A3, A4, T1, T2, T3, T4


def relaxation():
    X1, X2, X3, X4 = [], [], [], []
    A1, A2, A3, A4 = [], [], [], []
    T1, T2, T3, T4 = [], [], [], []

    fn1 = values['a12'] * x2 + values['a13'] * x3 + values['a14'] * x4
    fn2 = values['a21'] * x1 + values['a23'] * x3 + values['a24'] * x4
    fn3 = values['a31'] * x1 + values['a32'] * x2 + values['a34'] * x4
    fn4 = values['a31'] * x1 + values['a32'] * x2 + values['a33'] * x3

    b1, b2, b3, b4 = 0, 0, 0, 0
    for k in range(iter_num):
        if values['a11'] != 0:
            old1 = b1
            b1 = (values['b1'] - fn1.subs([(x2, values['x2']), (x3, values['x3']), (x4, values['x4'])])) / values['a11']
            values['s1'] = b1
            c1 = (values['l'] * values['s1']) + ((1 - values['l']) * values['x1'])
            values['x1'] = c1
            X1.append(c1)

            t1 = (abs(values['xt1'] - b1) / values['xt1']) * 100
            T1.append(t1)

            a1 = abs((b1 - old1) / b1) * 100 if k != 0 else '---'
            A1.append(a1)

        if values['a12'] != 0:
            old2 = b2
            b2 = (values['b2'] - fn2.subs([(x1, values['x1']), (x3, values['x3']), (x4, values['x4'])])) / values["a22"]
            values['s2'] = b2
            c2 = values['l'] * values['s2'] + (1 - values['l']) * values['x2']
            values['x2'] = c2
            X2.append(c2)

            t2 = (abs(values['xt2'] - b2) / values['xt2']) * 100
            T2.append(t2)

            a2 = abs((b2 - old2) / b2) * 100 if k != 0 else '---'
            A2.append(a2)

        if values['a13'] != 0:
            old3 = b3
            b3 = (values['b3'] - fn3.subs([(x1, values['x1']), (x2, values['x2']), (x4, values['x4'])])) / values["a33"]
            values['s3'] = b3
            c3 = values['l'] * values['s3'] + (1 - values['l']) * values['x3']
            values['x3'] = c3
            X3.append(c3)

            t3 = (abs(values['xt3'] - b3) / values['xt3']) * 100
            T3.append(t3)

            a3 = abs((b3 - old3) / b3) * 100 if k != 0 else '---'
            A3.append(a3)

        if values['a14'] != 0:
            old4 = b4
            b4 = (values['b4'] - fn4.subs([(x1, values['x1']), (x2, values['x2']), (x3, values['x3'])])) / values["a44"]
            values['s4'] = b4
            c4 = values['l'] * values['s4'] + (1 - values['l']) * values['x4']
            values['x4'] = c4
            X4.append(c4)

            t4 = (abs(values['xt4'] - b4) / values['xt4']) * 100
            T4.append(t4)

            a4 = abs((b4 - old4) / b4) * 100 if k != 0 else '---'
            A4.append(a4)

    return X1, X2, X3, X4, A1, A2, A3, A4, T1, T2, T3, T4


if st.button('go'):
    if checkbox_gaussian_seidel:
        _X1, _X2, _X3, _X4, _A1, _A2, _A3, _A4, _T1, _T2, _T3, _T4 = gaussian_seidel()

        data = {
            "X1": _X1,
            "X2": _X2,
            "X3": _X3,
            "A1": _A1,
            "A2": _A2,
            "A3": _A3,
            "T1": _T1,
            "T2": _T2,
            "T3": _T3,
        }
        if variable_num == 4:
            data["X4"] = _X4
            data["T4"] = _T4
            data["A4"] = _A4

        st.title('Gaussian Seidel Method')
        df = pd.DataFrame(data)
        df

    if checkbox_jacopi_method:
        _X1, _X2, _X3, _X4, _A1, _A2, _A3, _A4, _T1, _T2, _T3, _T4 = jacopi_method()

        data = {
            "X1": _X1,
            "X2": _X2,
            "X3": _X3,
            "A1": _A1,
            "A2": _A2,
            "A3": _A3,
            "T1": _T1,
            "T2": _T2,
            "T3": _T3,
        }

        if variable_num == 4:
            data["X4"] = _X4
            data["T4"] = _T4
            data["A4"] = _A4

        st.title('Jacobi Method')
        df = pd.DataFrame(data)
        df

    if checkbox_relaxation:
        _X1, _X2, _X3, _X4, _A1, _A2, _A3, _A4, _T1, _T2, _T3, _T4 = relaxation()

        data = {
            "X1": _X1,
            "X2": _X2,
            "X3": _X3,
            "A1": _A1,
            "A2": _A2,
            "A3": _A3,
            "T1": _T1,
            "T2": _T2,
            "T3": _T3,
        }

        if variable_num == 4:
            data["X4"] = _X4
            data["T4"] = _T4
            data["A4"] = _A4

        st.title('Over Relaxation Method')
        df = pd.DataFrame(data)
        df
