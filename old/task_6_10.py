import random
import numpy as np
import sympy
import matplotlib.pyplot as plt
from sympy import symbols, lambdify

coff_values = {
    'a11': 0, 'a12': 0, 'a13': 0, 'a14': 0,
    'a21': 0, 'a22': 0, 'a23': 0, 'a24': 0,
    'a31': 0, 'a32': 0, 'a33': 0, 'a34': 0,
}

x_values = {
    'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0,
}

b_values = {
    'b1': 0, 'b2': 0, 'b3': 0
}

linear_combination_factors = [
    {'l': 0, 'r': 0},
    {'l': 0, 'r': 0},
    {'l': 0, 'r': 0},
]


def get_random_value():
    random_number = random.randint(-2, 2)
    return random_number


def build_table(data):
    table_header = ''
    table_body = ''
    for key, value in data.items():
        table_header += '%-15s' % key
        table_body += '%-15s' % value
    return table_header, table_body


for coff in coff_values.keys():
    while coff_values[coff] == 0:
        coff_values[coff] = get_random_value()

for x_value in x_values.keys():
    x_values[x_value] = get_random_value()

for index, b_value in enumerate(b_values.keys()):
    b_values[b_value] = coff_values[f'a{index+1}1'] * x_values['x1'] + coff_values[f'a{index+1}2'] * x_values['x2'] + coff_values[f'a{index+1}3'] * x_values['x3'] + coff_values[f'a{index+1}4'] * x_values['x4']

for equation_factors in linear_combination_factors:
    while equation_factors['l'] == equation_factors['r']:
        equation_factors['l'] = get_random_value()
        equation_factors['r'] = get_random_value()

for i in range(1, 5):
    coff_values[f'a4{i}'] = linear_combination_factors[0]['l'] * coff_values[f'a1{i}'] + linear_combination_factors[1]['l'] * coff_values[f'a2{i}'] + linear_combination_factors[2]['l'] * coff_values[f'a3{i}']

b_values['b4'] = linear_combination_factors[0]['r'] * b_values['b1'] + linear_combination_factors[1]['r'] * b_values['b2'] + linear_combination_factors[2]['r'] * b_values['b3']

header, body = build_table(coff_values)
print(header)
print(body)

header, body = build_table(x_values)
print(header)
print(body)

header, body = build_table(b_values)
print(header)
print(body)

x1 = symbols('x1')
x2 = symbols('x2')
x3 = symbols('x3')
x4 = symbols('x4')

f_1 = coff_values['a11'] * x1 + coff_values['a12'] * x2 + coff_values['a13'] * x3 + coff_values['a14'] * x4
f_2 = coff_values['a21'] * x1 + coff_values['a22'] * x2 + coff_values['a23'] * x3 + coff_values['a24'] * x4
f_3 = coff_values['a31'] * x1 + coff_values['a32'] * x2 + coff_values['a33'] * x3 + coff_values['a34'] * x4
f_4 = coff_values['a41'] * x1 + coff_values['a42'] * x2 + coff_values['a43'] * x3 + coff_values['a44'] * x4

proove = f"E1 right_side * {linear_combination_factors[0]['r']}, E1 left_side * {linear_combination_factors[0]['l']}\n"
proove += f"E2 right_side * {linear_combination_factors[1]['r']}, E2 left_side * {linear_combination_factors[1]['l']}\n"
proove += f"E3 right_side * {linear_combination_factors[2]['r']}, E3 left_side * {linear_combination_factors[2]['l']}\n"
proove += 'so its singular system with no solution'
print(proove)

print(f'{f_1} = {b_values["b1"]}')
print(f'{f_2} = {b_values["b2"]}')
print(f'{f_3} = {b_values["b3"]}')
print(f'{f_4} = {b_values["b4"]}')

print(f'{f_1} - {b_values["b1"]},')
print(f'{f_2} - {b_values["b2"]},')
print(f'{f_3} - {b_values["b3"]},')
print(f'{f_4} - {b_values["b4"]}')

