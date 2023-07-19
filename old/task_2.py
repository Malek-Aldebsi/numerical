import numpy as np
import sympy
import matplotlib.pyplot as plt
import math
import random

from sympy import diff, symbols, factorial, exp, pretty_print, sin, lambdify


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
    'x': [-5, 5]
}

coefficientsValues = {}
for coff, coff_range in ranges.items():
    random_num = get_random_value(coff_range[0], coff_range[1])
    coefficientsValues[coff] = random_num


def my_function(x):
    return coefficientsValues['a1'] * np.exp(coefficientsValues['a2'] * x) * np.sin(coefficientsValues['a3'] * x + coefficientsValues['a4'])
    # return 0.6 - x + 2 * x**2 - x**3 + 0.25 * x**4


x = coefficientsValues.pop('x')
# x = 1

header = ''
values = ''
for i in range(1, len(ranges)):
    header += "%-15s" % f'a{i}'
    values += "%-15s" % (coefficientsValues[f'a{i}'])

exact_solution = my_function(x)

header += "%-15s%-15s" % ('x', 'f(x)')
values += "%-15s%-15s" % (x, format(exact_solution, ".16g"))

print(header)
print(values)
print('-' * 170)

a = get_random_value(x-1, x+1)
# a = 0
a_solution = my_function(a)

x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
y_values = my_function(x_values)

plt.plot(x_values, y_values)
plt.plot(x, exact_solution, 'ro', label='Data Point')
plt.plot(a, a_solution, 'bo', label='Data Point')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.text(x, exact_solution, f'({x}, {exact_solution})', verticalalignment='bottom', horizontalalignment='left')
plt.text(a, a_solution, f'({a}, {a_solution})', verticalalignment='bottom', horizontalalignment='left')
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Function')
plt.show()


y = symbols("y")
f_x = coefficientsValues["a1"] * exp(coefficientsValues["a2"] * y) * sin(
    coefficientsValues["a3"] * y + coefficientsValues["a4"])
# f_x = 0.6 - y + 2 * y**2 - y**3 + 0.25 * y**4

num_of_terms = 8
result = 0

approximate_values = []
for iteration in range(0, num_of_terms):
    result += diff(f_x, y, iteration).subs(y, a) * (y - a) ** iteration / factorial(iteration)
    approximate_values.append(result.subs(y, x))

header = "%-25s" % f''
_approximate_values = "%-25s" % f'approximate values'
_absolute_truncation_error = "%-25s" % f'abs truncation error'
_relative_percentage_true_error = "%-25s" % f'relative % true error'
for i in range(0, len(approximate_values)):
    header += "%-25s" % f'# of terms = {i + 1}'
    _approximate_values += "%-25s" % (approximate_values[i])
    _absolute_truncation_error += "%-25s" % (abs(exact_solution - approximate_values[i]))
    _relative_percentage_true_error += "%-25s" % (abs((exact_solution - approximate_values[i])/exact_solution))

print(f'exact solution: {exact_solution}')
print(header)
print(_approximate_values)
print(_absolute_truncation_error)
print(_relative_percentage_true_error)



