import streamlit as st
from sympy import sympify
import pandas as pd
from utils import bi_section, false_position, fixed_point, newton_raphson, secant, \
    draw_approximate_and_true_errors_for_single_method, draw_true_errors_for_all_methods, \
    draw_approximate_errors_for_all_methods, get_bisection_doc, get_false_position_doc, get_fixed_point_doc, \
    get_newton_raphson_doc, get_secant_doc

examples = [
    {
        'function': r'4.84 * x**3 + 3.59 * x**2 + 2.86 * x + 0.134',
        'g(x)': r'4.84 * x**3 + 3.59 * x**2 + 3.86 * x + 0.134',
        'lower': '-1',
        'upper': '0',
        'initial': '0',
        'initial_1': '1',
        'exact': '-0.05',
        'app_error': '0.005',
        'true_error': '0.05',
        'iter_num': '10',
     },
]
example_index = 0

st.title('Methods')
f_x = st.text_input("Enter your function", value=examples[example_index]['function'])
f_x = sympify(f_x)
st.latex(f_x)

checkbox_bi_section = st.checkbox('Bi Section')
if checkbox_bi_section:
    col1, col2 = st.columns(2)
    with col1:
        bi_section_lower = float(st.text_input('X lower (BiSection)', value=examples[example_index]['lower']))
    with col2:
        bi_section_upper = float(st.text_input('X upper (BiSection)', value=examples[example_index]['upper']))

checkbox_false_position = st.checkbox('False Position')
if checkbox_false_position:
    col1, col2 = st.columns(2)
    with col1:
        false_position_lower = float(st.text_input('X lower (FalsePosition)', value=examples[example_index]['lower']))
    with col2:
        false_position_upper = float(st.text_input('X upper (FalsePosition)', value=examples[example_index]['upper']))

checkbox_fixed_point = st.checkbox('Fixed point')
if checkbox_fixed_point:
    col1, col2 = st.columns(2)
    with col1:
        g_x = st.text_input("Enter g(x)", value=examples[example_index]['g(x)'])
        g_x = sympify(g_x)
        st.latex(g_x)
    with col2:
        fixed_point_initial = float(st.text_input('X initial (FixedPoint)', value=examples[example_index]['initial']))

checkbox_newton_raphson = st.checkbox('Newton Raphson')
if checkbox_newton_raphson:
    newton_raphson_initial = float(st.text_input('X initial (NewtonRaphson)', value=examples[example_index]['initial']))

checkbox_secant = st.checkbox('Secant')
if checkbox_secant:
    col1, col2 = st.columns(2)
    with col1:
        secant_initial = float(st.text_input('X initial (Secant)', value=examples[example_index]['initial']))
    with col2:
        secant_initial1 = float(st.text_input('X1 initial (Secant)', value=examples[example_index]['initial_1']))

st.markdown("---")

st.title('Conditions')

col1, col2, col3 = st.columns(3)
with col1:
    checkbox_number_of_iterations = st.checkbox('Number ot iteration')
    if checkbox_number_of_iterations:
        number_of_iterations = int(st.text_input('Number ot iteration', value=10))

with col2:
    checkbox_true_relative_error_cond = st.checkbox('True Relative Error')
    if checkbox_true_relative_error_cond:
        true_relative_error_cond = float(st.text_input('True Relative Error Cond', value=0.005)) * 100
        x_exact = float(st.text_input("Enter your exact solution", value=-0.05))

with col3:
    checkbox_approximate_relative_error_cond = st.checkbox('Approximate Relative Error')
    if checkbox_approximate_relative_error_cond:
        approximate_relative_error_cond = float(st.text_input('Approximate Relative Error Cond', value=0.005)) * 100

if st.button('Go', use_container_width=True):
    bi_section_errors = None
    false_position_errors = None
    fixed_point_errors = None
    newton_raphson_errors = None
    secant_errors = None

    if checkbox_bi_section:
        bi_section_data, bi_section_errors = bi_section(f_x, x_exact if checkbox_true_relative_error_cond else None, bi_section_lower, bi_section_upper, approximate_relative_error_cond if checkbox_approximate_relative_error_cond else None, true_relative_error_cond if checkbox_true_relative_error_cond else None, number_of_iterations if checkbox_number_of_iterations else None)
        st.title('BiSection Method')
        bi_section_table = pd.DataFrame(bi_section_data)
        bi_section_table

        draw_approximate_and_true_errors_for_single_method('Bi Section', bi_section_errors)
        get_bisection_doc(f_x, number_of_iterations if checkbox_number_of_iterations else 'undetermined', approximate_relative_error_cond if checkbox_approximate_relative_error_cond else 'undetermined', x_exact if checkbox_true_relative_error_cond else 'unKnown', bi_section_data)
        st.download_button("Download Document", data=open('BiSection.docx', "rb"), file_name="BiSection.docx")

    if checkbox_false_position:
        false_position_data, false_position_errors = false_position(f_x, x_exact if checkbox_true_relative_error_cond else None, false_position_lower, false_position_upper, approximate_relative_error_cond if checkbox_approximate_relative_error_cond else None, true_relative_error_cond if checkbox_true_relative_error_cond else None, number_of_iterations if checkbox_number_of_iterations else None)
        st.title('FalsePosition Method')
        false_position_table = pd.DataFrame(false_position_data)
        false_position_table

        draw_approximate_and_true_errors_for_single_method('False Position', false_position_errors)
        get_false_position_doc(f_x, number_of_iterations if checkbox_number_of_iterations else 'undetermined', approximate_relative_error_cond if checkbox_approximate_relative_error_cond else 'undetermined', x_exact if checkbox_true_relative_error_cond else 'unKnown', false_position_data)
        st.download_button("Download Document", data=open('False_Position.docx', "rb"), file_name="False_Position.docx")

    if checkbox_fixed_point:
        fixed_point_data, fixed_point_errors = fixed_point(f_x, g_x, x_exact if checkbox_true_relative_error_cond else None, fixed_point_initial, approximate_relative_error_cond if checkbox_approximate_relative_error_cond else None, true_relative_error_cond if checkbox_true_relative_error_cond else None, number_of_iterations if checkbox_number_of_iterations else None)
        st.title('FixedPoint Method')
        fixed_point_table = pd.DataFrame(fixed_point_data)
        fixed_point_table

        draw_approximate_and_true_errors_for_single_method('Fixed Point', fixed_point_errors)
        get_fixed_point_doc(f_x, number_of_iterations if checkbox_number_of_iterations else 'undetermined', approximate_relative_error_cond if checkbox_approximate_relative_error_cond else 'undetermined', fixed_point_data)
        st.download_button("Download Document", data=open('FixedPoint.docx', "rb"), file_name="FixedPoint.docx")

    if checkbox_newton_raphson:
        newton_raphson_data, newton_raphson_errors = newton_raphson(f_x, x_exact if checkbox_true_relative_error_cond else None, newton_raphson_initial, approximate_relative_error_cond if checkbox_approximate_relative_error_cond else None, true_relative_error_cond if checkbox_true_relative_error_cond else None, number_of_iterations if checkbox_number_of_iterations else None)
        st.title('Newton Raphson')
        newton_raphson_table = pd.DataFrame(newton_raphson_data)
        newton_raphson_table

        draw_approximate_and_true_errors_for_single_method('Newton Raphson', newton_raphson_errors)
        get_newton_raphson_doc(f_x, number_of_iterations if checkbox_number_of_iterations else 'undetermined', approximate_relative_error_cond if checkbox_approximate_relative_error_cond else 'undetermined', newton_raphson_data)
        st.download_button("Download Document", data=open('Newton Raphson.docx', "rb"), file_name="Newton Raphson.docx")

    if checkbox_secant:
        secant_data, secant_errors = secant(f_x, x_exact if checkbox_true_relative_error_cond else None, secant_initial, secant_initial1, approximate_relative_error_cond if checkbox_approximate_relative_error_cond else None, true_relative_error_cond if checkbox_true_relative_error_cond else None, number_of_iterations if checkbox_number_of_iterations else None)
        st.title('Secant')
        secant_table = pd.DataFrame(secant_data)
        secant_table

        draw_approximate_and_true_errors_for_single_method('Secant', secant_errors)
        get_secant_doc(f_x, number_of_iterations if checkbox_number_of_iterations else 'undetermined',
                           approximate_relative_error_cond if checkbox_approximate_relative_error_cond else 'undetermined',
                           secant_data)
        st.download_button("Download Document", data=open('Secant.docx', "rb"), file_name="Secant.docx")

    if (checkbox_bi_section + checkbox_false_position + checkbox_fixed_point + checkbox_newton_raphson + checkbox_secant > 1) and (checkbox_true_relative_error_cond):
        st.title('True Errors Comparison')
        draw_true_errors_for_all_methods(bi_section_errors, false_position_errors, fixed_point_errors, newton_raphson_errors, secant_errors)

    if (checkbox_bi_section + checkbox_false_position + checkbox_fixed_point + checkbox_newton_raphson + checkbox_secant > 1) and (checkbox_approximate_relative_error_cond):
        st.title('Approximate Errors Comparison')
        draw_approximate_errors_for_all_methods(bi_section_errors, false_position_errors, fixed_point_errors, newton_raphson_errors, secant_errors)
