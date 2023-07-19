import math
import random
import numpy as np
import matplotlib.pyplot as plt


def chopping(number, sf):
    number = str(number)
    number = list(number)
    nz_flag = False
    counter = 0
    for i, ch in enumerate(number):
        if ch not in ['0', '.']:
            nz_flag = True
        if nz_flag and ch != '.':
            counter += 1
        if counter > sf and ch != '.':
            number[i] = '0'
    number = ''.join(number)
    number = float(number)
    return number


def get_random_value(lower_bound, upper_bound):
    random_number = random.uniform(lower_bound, upper_bound)
    if random_number == 0:
        random_number = get_random_value()
    return float(format(random_number, ".3g"))


ranges = {
    'a1': [-2, 2],
    'a2': [-2, 2],
    'a3': [-2, 2],
    'a4': [-math.pi / 2, math.pi / 2],
    'a5': [-2, 2],
    'a6': [-100, 100],
    'a7': [-10, 10],
    'a8': [-2, 2],
    'a9': [-1, 1],
    'x': [-5, 5]
}


coefficientsValues = {}
for coff, coff_range in ranges.items():
    random_num = get_random_value(coff_range[0], coff_range[1])
    coefficientsValues[coff] = random_num

x = coefficientsValues.pop('x')
while coefficientsValues['a6'] * x < 0:
    x = get_random_value(ranges['x'][0], ranges['x'][1])

header = ''
values = ''
for i in range(1, len(ranges)):
    header += "%-15s" % f'a{i}'
    values += "%-15s" % (coefficientsValues[f'a{i}'])

exact_solution = coefficientsValues['a1'] * np.exp(coefficientsValues['a2'] * x) * np.sin(coefficientsValues['a3'] * x + coefficientsValues['a4']) + coefficientsValues['a5'] * np.log(coefficientsValues['a6'] * x) + coefficientsValues['a7'] * x + coefficientsValues['a8'] * x ** 2 + coefficientsValues['a9'] * x ** 3

header += "%-15s%-15s" % ('x', 'f(x)')
values += "%-15s%-15s" % (x, format(exact_solution, ".16g"))

print(header)
print(values)
print('-' * 170)


def my_function(x):
    return coefficientsValues['a1'] * np.exp(coefficientsValues['a2'] * x) * np.sin(coefficientsValues['a3'] * x + coefficientsValues['a4']) + coefficientsValues['a5'] * np.log(coefficientsValues['a6'] * x) + coefficientsValues['a7'] * x + coefficientsValues['a8'] * x ** 2 + coefficientsValues['a9'] * x ** 3


x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
y_values = my_function(x_values)

plt.plot(x_values, y_values)
plt.plot(x, exact_solution, 'ro', label='Data Point')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.text(x, exact_solution, f'({x}, {exact_solution})', verticalalignment='bottom', horizontalalignment='left')
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Function')
plt.show()


def f_x(sig_num):
    f_x = 'a1 * e^(a2 * x) * sin(a3 * x + a4) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3'
    _1_operation = chopping(coefficientsValues['a2'] * x, sig_num)
    _1_statement = f"a1 * e^({_1_operation}) * sin(a3 * x + a4) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _2_operation = chopping(math.exp(_1_operation), sig_num)
    _2_statement = f"a1 * {_2_operation} * sin(a3 * x + a4) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _3_operation = chopping(coefficientsValues['a1']*_2_operation, sig_num)
    _3_statement = f"{_3_operation} * sin(a3 * x + a4) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _4_operation = chopping(coefficientsValues['a3'] * x, sig_num)
    _4_statement = f"{_3_operation} * sin({_4_operation} + a4) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _5_operation = chopping(_4_operation + coefficientsValues['a4'], sig_num)
    _5_statement = f"{_3_operation} * sin({_5_operation}) + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _6_operation = chopping(math.sin(_5_operation), sig_num)
    _6_statement = f"{_3_operation} * {_6_operation} + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _7_operation = chopping(_3_operation * _6_operation, sig_num)
    _7_statement = f"{_7_operation} + a5 * log(a6 * x) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _8_operation = chopping(coefficientsValues['a6'] * x, sig_num)
    _8_statement = f"{_7_operation} + a5 * log({_8_operation}) + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _9_operation = chopping(math.log(_8_operation), sig_num)
    _9_statement = f"{_7_operation} + a5 * {_9_operation} + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _10_operation = chopping(coefficientsValues['a5'] * _9_operation, sig_num)
    _10_statement = f"{_7_operation} + {_10_operation} + a7 * x + a8 * x ^ 2 + a9 * x^3"
    _11_operation = chopping(coefficientsValues['a7'] * x, sig_num)
    _11_statement = f"{_7_operation} + {_10_operation} + {_11_operation} + a8 * x ^ 2 + a9 * x^3"
    _12_operation = chopping(x**2, sig_num)
    _12_statement = f"{_7_operation} + {_10_operation} + {_11_operation} + a8 * {_12_operation} + a9 * x^3"
    _13_operation = chopping(coefficientsValues['a8']*_12_operation, sig_num)
    _13_statement = f"{_7_operation} + {_10_operation} + {_11_operation} + {_13_operation} + a9 * x^3"
    _14_operation = chopping(x ** 3, sig_num)
    _14_statement = f"{_7_operation} + {_10_operation} + {_11_operation} + {_13_operation} + a9 * {_14_operation}"
    _15_operation = chopping(coefficientsValues['a9'] * _14_operation, sig_num)
    _15_statement = f"{_7_operation} + {_10_operation} + {_11_operation} + {_13_operation} + {_15_operation}"
    _16_operation = chopping(_7_operation + _10_operation, sig_num)
    _16_statement = f"{_16_operation} + {_11_operation} + {_13_operation} + {_15_operation}"
    _17_operation = chopping(_16_operation + _11_operation, sig_num)
    _17_statement = f"{_17_operation} + {_13_operation} + {_15_operation}"
    _18_operation = chopping(_17_operation + _13_operation, sig_num)
    _18_statement = f"{_18_operation} + {_15_operation}"
    _19_operation = chopping(_18_operation + _15_operation, sig_num)
    _19_statement = f"{_19_operation}"

    print(f_x + '\n' + _1_statement, '\n', _2_statement, '\n', _3_statement, '\n', _4_statement, '\n', _5_statement, '\n', _6_statement, '\n', _7_statement, '\n', _8_statement, '\n', _9_statement, '\n', _10_statement + '\n' + _11_statement, '\n', _12_statement, '\n', _13_statement, '\n', _14_statement, '\n', _15_statement, '\n', _16_statement, '\n', _17_statement, '\n', _18_statement, '\n', _19_statement)
    return _19_operation


print('3 significant figures:')
_3_significant_figures = f_x(3)
print('5 significant figures:')
_5_significant_figures = f_x(5)
print('7 significant figures:')
_7_significant_figures = f_x(7)
print('9 significant figures:')
_9_significant_figures = f_x(9)

print(f'exact solution: {exact_solution}')
print(f'approximate solution with 3 significant: {_3_significant_figures}')
print(f'approximate solution with 5 significant: {_5_significant_figures}')
print(f'approximate solution with 7 significant: {_7_significant_figures}')
print(f'approximate solution with 9 significant: {_9_significant_figures}')
print(f'absolute round off error with 3 significant: {abs(float(exact_solution)-_3_significant_figures)}')
print(f'absolute round off error with 5 significant: {abs(float(exact_solution)-_5_significant_figures)}')
print(f'absolute round off error with 7 significant: {abs(float(exact_solution)-_7_significant_figures)}')
print(f'absolute round off error with 9 significant: {abs(float(exact_solution)-_9_significant_figures)}')
print(f'relative percentage round off error with 3 significant: {abs(float(exact_solution)-_3_significant_figures/float(exact_solution))}')
print(f'relative percentage round off error with 5 significant: {abs(float(exact_solution)-_5_significant_figures/float(exact_solution))}')
print(f'relative percentage round off error with 7 significant: {abs(float(exact_solution)-_7_significant_figures/float(exact_solution))}')
print(f'relative percentage round off error with 9 significant: {abs(float(exact_solution)-_9_significant_figures/float(exact_solution))}')
