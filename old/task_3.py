import numpy as np
import sympy
import matplotlib.pyplot as plt
import math
import random

from sympy import diff, symbols, cos, pi, ln, sin, lambdify


def get_random_value(lower_bound, upper_bound):
    random_number = random.uniform(lower_bound, upper_bound)
    if random_number == 0:
        random_number = get_random_value()
    return float(format(random_number, ".3g"))


ranges = {
    'a7': [-10, 10],
    'a8': [-2, 2],
    'a9': [-1, 2],

    'x_well': [-5, 5],
    'x_ill': [-5, 5]
}

coefficientsValues = {}
x_well_condition_num = 1
x_ill_condition_num = 1

while x_well_condition_num > 0.25 or x_ill_condition_num < 10:
    for coff, coff_range in ranges.items():
        random_num = get_random_value(coff_range[0], coff_range[1])
        coefficientsValues[coff] = random_num
    coefficientsValues.pop('x_well')
    coefficientsValues.pop('x_ill')

    y = symbols("y")
    f_x = coefficientsValues["a7"] * y + coefficientsValues["a8"] * y**2 + coefficientsValues["a9"] * y**3

    x_well = get_random_value(ranges['x_well'][0], ranges['x_well'][1])
    f_x_well = f_x.subs(y, x_well)
    f_x_well_d_x = diff(f_x, y, 1).subs(y, x_well)
    x_well_condition_num = abs((f_x_well_d_x * x_well) / f_x_well)
    print(x_well_condition_num)

    x_ill = get_random_value(ranges['x_ill'][0], ranges['x_ill'][1])
    f_x_ill = f_x.subs(y, x_ill)
    f_x_ill_d_x = diff(f_x, y, 1).subs(y, x_ill)
    x_ill_condition_num = abs((f_x_ill_d_x * x_ill) / f_x_ill)
    print(x_ill_condition_num)


# y = symbols("y")
# f_x = ln(y)
#
# x_well = 1.1
# f_x_well = f_x.subs(y, x_well)
# f_x_well_d_x = diff(f_x, y, 1).subs(y, x_well)
# x_well_condition_num = abs((f_x_well_d_x * x_well) / f_x_well)
# print(x_well_condition_num)
#
# x_ill = 1.01
# f_x_ill = f_x.subs(y, x_ill)
# f_x_ill_d_x = diff(f_x, y, 1).subs(y, x_ill)
# x_ill_condition_num = abs((f_x_ill_d_x * x_ill) / f_x_ill)
# print(x_ill_condition_num)


header = ''
values = ''
for i in range(0, len(coefficientsValues)):
    header += "%-25s" % f'a{i + 7}'
    values += "%-25s" % (coefficientsValues[f'a{i + 7}'])

header += "%-25s%-25s%-25s%-25s" % ('x_well', 'x_ill', 'f(x_well)', 'f(x_ill)')
values += "%-25s%-25s%-25s%-25s" % (x_well, x_ill, f_x_well, f_x_ill)

print(header)
print(values)

f_x_for_numpy = lambdify(y, f_x, modules=['numpy'])
x_values = np.linspace(ranges['x_well'][0], ranges['x_well'][1], 100)
y_values = f_x_for_numpy(x_values)

plt.plot(x_values, y_values)
plt.plot(x_well, f_x_well, 'ro', label='Data Point')
plt.plot(x_ill, f_x_ill, 'bo', label='Data Point')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.text(x_well, f_x_well, f'well point: ({x_well}, {f_x_well})',
         verticalalignment='bottom', horizontalalignment='left')
plt.text(x_ill, f_x_ill, f'ill point: ({x_ill}, {f_x_ill})',
         verticalalignment='bottom', horizontalalignment='left')
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Function')
plt.show()

print("f(x)= " + str(coefficientsValues["a7"]) + " * x + " + str(coefficientsValues["a8"]) + " * x^2 + " + str(coefficientsValues["a9"]) + " * x^3")
print("diff(f(x))= " + str(coefficientsValues["a7"]) + " " + str(coefficientsValues["a8"]) + " * 2 * x " + str(coefficientsValues["a9"]) + " * 3 * x^2")
print("ConditionNumber = diff(f(x)) * x / f(x)")

print('for x_well:')
print("f(" + str(x_well) + ") = " + str(f_x_well))
print("diff(f(" + str(x_well) + "))= " + str(f_x_well_d_x))
print("Well Condition Number= " + str(x_well_condition_num))
print('for x_ill')
print("diff(f(" + str(x_ill) + "))= " + str(f_x_ill_d_x))
print("f(" + str(x_ill) + ") = " + str(f_x_ill))
print("Ill Condition Number= " + str(x_ill_condition_num))
