import numpy as np
from sympy import diff, symbols, ln, exp, pretty_print, sin, lambdify, log
import matplotlib.pyplot as plt
import math
import random


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
    'x': [-5, 5]
}

coefficientsValues = {}
for coff, coff_range in ranges.items():
    random_num = get_random_value(coff_range[0], coff_range[1])
    coefficientsValues[coff] = random_num

x = coefficientsValues.pop('x')
while coefficientsValues['a6'] * x < 0:
    x = get_random_value(ranges['x'][0], ranges['x'][1])


y = symbols("y")
f_x = coefficientsValues["a1"] * exp(coefficientsValues["a2"] * y) * sin(
    coefficientsValues["a3"] * y + coefficientsValues["a4"]) + coefficientsValues['a5'] * log(
    coefficientsValues['a6'] * y) + coefficientsValues['a7'] * y + coefficientsValues['a8'] * y ** 2
f_x_r_s = f_x.subs(y, x)

f_x = coefficientsValues["a1"] * exp(coefficientsValues["a2"] * y) * sin(
    coefficientsValues["a3"] * y + coefficientsValues["a4"]) + coefficientsValues['a5'] * log(
    coefficientsValues['a6'] * y) + coefficientsValues['a7'] * y + coefficientsValues['a8'] * y ** 2 - f_x_r_s

f_x_exact = f_x.subs(y, x)

header = ''
str_coff = ''
for i in range(1, 9):
    header += "%-15s" % f'a{i}'
    str_coff += "%-15s" % (coefficientsValues[f'a{i}'])

header += "%-15s%-15s" % ('x', 'f(x)')
str_coff += "%-15s%-15s" % (x, f_x_exact)

print(header)
print(str_coff)
print('-' * 170)

f_x_for_numpy = lambdify(y, f_x, modules=['numpy'])
x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
y_values = f_x_for_numpy(x_values)

plt.plot(x_values, y_values)
plt.plot(x, f_x_exact, 'ro', label='Data Point')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.text(x, f_x_exact, f'({x}, {f_x_exact})', verticalalignment='bottom', horizontalalignment='left')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('My Function')

plt.show()

print('(true_relative_error > 0.005 and approximate_relative_error > 0.005) or iteration num < 10')


def bracketing_methods_initial_guesses():
    x_l = get_random_value(ranges['x'][0], x)
    while coefficientsValues['a6'] * x_l < 0:
        x_l = get_random_value(ranges['x'][0], x)

    x_u = get_random_value(x, ranges['x'][1])
    while coefficientsValues['a6'] * x_u < 0:
        x_u = get_random_value(x, ranges['x'][1])
    return x_l, x_u


def bi_section(x, x_l, x_u, f_x):
    print('BiSection Method: ')
    f_x_l = f_x.subs(y, x_l)
    f_x_u = f_x.subs(y, x_u)

    while f_x_u * f_x_l > 0:
        x_u = get_random_value(x, ranges['x'][1])
        while coefficientsValues['a6'] * x_u < 0:
            x_u = get_random_value(x, ranges['x'][1])
        f_x_u = f_x.subs(y, x_u)

        if f_x_u * f_x_l > 0:
            x_l = get_random_value(ranges['x'][0], x)
            while coefficientsValues['a6'] * x_l < 0:
                x_l = get_random_value(ranges['x'][0], x)
            f_x_l = f_x.subs(y, x_l)

    # x = -0.05
    # f_x = 4.84 * y**3 + 3.59 * y**2 + 2.86 * y + 0.134
    #
    # x_l = -1
    # f_x_l = f_x.subs(y, x_l)
    #
    # x_u = 0
    # f_x_u = f_x.subs(y, x_u)

    x_r = (x_u + x_l) / 2
    f_x_r = f_x.subs(y, x_r)

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'Xl', 'Xu', 'Xr', 'f(Xr)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x - x_r) / x) * 100
    approximate_relative_error = 1
    errors = {}
    while (true_relative_error > 0.005 and approximate_relative_error > 0.005) or iter_num <= 10:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_l, x_u, x_r, f_x_r, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        if f_x_r * f_x_u < 0:
            x_l = x_r
            f_x_l = f_x_r
        else:
            x_u = x_r
            f_x_u = f_x_r

        pre_x_r = x_r
        x_r = (x_u + x_l) / 2
        f_x_r = f_x.subs(y, x_r)

        iter_num += 1
        true_relative_error = abs((x - x_r) / x) * 100
        approximate_relative_error = abs((x_r - pre_x_r) / x_r) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]
    print(header)
    print(values)

    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for BiSection method')

    plt.show()

    return errors


def false_position(x, x_l, x_u, f_x):
    print('False Position Method: ')
    f_x_l = f_x.subs(y, x_l)
    f_x_u = f_x.subs(y, x_u)

    while f_x_u * f_x_l > 0:
        x_u = get_random_value(x, ranges['x'][1])
        while coefficientsValues['a6'] * x_u < 0:
            x_u = get_random_value(x, ranges['x'][1])
        f_x_u = f_x.subs(y, x_u)

        if f_x_u * f_x_l > 0:
            x_l = get_random_value(ranges['x'][0], x)
            while coefficientsValues['a6'] * x_l < 0:
                x_l = get_random_value(ranges['x'][0], x)
            f_x_l = f_x.subs(y, x_l)

    # x = -0.05
    # f_x = 4.84 * y**3 + 3.59 * y**2 + 2.86 * y + 0.134
    #
    # x_l = -0.5
    # f_x_l = f_x.subs(y, x_l)
    #
    # x_u = 0.5
    # f_x_u = f_x.subs(y, x_u)

    x_r = x_u - f_x_u * (x_u - x_l) / (f_x_u-f_x_l)
    f_x_r = f_x.subs(y, x_r)

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'Xl', 'Xu', 'Xr', 'f(Xr)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x - x_r) / x) * 100
    approximate_relative_error = 1
    errors = {}
    while (true_relative_error > 0.005 and approximate_relative_error > 0.005) or iter_num <= 10:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_l, x_u, x_r, f_x_r, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        if f_x_r * f_x_u < 0:
            x_l = x_r
            f_x_l = f_x_r
        else:
            x_u = x_r
            f_x_u = f_x_r

        pre_x_r = x_r
        x_r = x_u - f_x_u * (x_u - x_l) / (f_x_u-f_x_l)
        f_x_r = f_x.subs(y, x_r)

        iter_num += 1
        true_relative_error = abs((x - x_r) / x) * 100
        approximate_relative_error = abs((x_r - pre_x_r) / x_r) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]
    print(header)
    print(values)

    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for False Position method')

    plt.show()

    return errors


def open_methods_initial_guesses():
    x_i = get_random_value(ranges['x'][0], ranges['x'][1])
    while coefficientsValues['a6'] * x_i < 0:
        x_i = get_random_value(ranges['x'][0], ranges['x'][1])

    x_i_1 = get_random_value(ranges['x'][0], ranges['x'][1])
    while coefficientsValues['a6'] * x_i_1 < 0:
        x_i_1 = get_random_value(ranges['x'][0], ranges['x'][1])

    return x_i, x_i_1


def fixed_point(x, x_i, f_x):
    print('Fixed Point Method: ')

    g_x_1 = f_x + y  # +x
    g_x_2 = (f_x - coefficientsValues['a7'] * y) / -coefficientsValues['a7']  # move x
    g_x_3 = ((f_x - coefficientsValues['a8'] * y ** 2) / -coefficientsValues['a8']) ** (1/2)  # move x^2
    g_x_4 = (10 ** ((f_x - coefficientsValues['a5'] * log(coefficientsValues['a6'] * y)) / -coefficientsValues['a5'])) / coefficientsValues['a6']  # move log
    g_x_list = [g_x_1, g_x_2, g_x_3, g_x_4]

    k = 1
    g_x_iter = 0
    while k >= 0.3 and g_x_iter < len(g_x_list):
        g_x = g_x_list[g_x_iter]
        g_x_d_x = diff(g_x, y, 1)
        k = abs(g_x_d_x.subs(y, x))
        g_x_iter += 1
    if k > 0.3:
        exit()

    g_x_i = g_x.subs(y, x_i)
    f_x_i = f_x.subs(y, x_i)

    # x = 1.19169
    # f_x = -5 * y**5 + y**4 + 10
    # g_x = ((y**4 + 10) / 5)**(1 / 5)
    #
    # x_i = 2
    # g_x_i = g_x.subs(y, x_i)
    # f_x_i = f_x.subs(y, x_i)

    header = "%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'x', 'f(x)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x - x_i) / x) * 100
    approximate_relative_error = 1
    errors = {}
    while (true_relative_error > 0.005 and approximate_relative_error > 0.005) or iter_num <= 10:
        values += "%-25s%-25s%-25s%-25s%-25s\n" % (
        iter_num, x_i, f_x_i, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        pre_x_i = x_i
        x_i = g_x_i
        g_x_i = g_x.subs(y, x_i)
        f_x_i = f_x.subs(y, x_i)

        iter_num += 1
        true_relative_error = abs((x - x_i) / x) * 100
        approximate_relative_error = abs((x_i - pre_x_i) / x_i) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]
    print(header)
    print(values)

    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for Fixed Point method')

    plt.show()

    return errors


def newton_raphson(x, x_i, f_x):
    print('Newton Raphson Method: ')

    # x = -1.21
    # f_x = 2.46 * ln(-3.45 * y) - 0.923 * y - 4.64
    #
    # x_i = -0.1

    f_x_i = f_x.subs(y, x_i)
    x_i_1 = x_i - f_x.subs(y, x_i) / diff(f_x, y, 1).subs(y, x_i)

    header = "%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'x', 'f(x)', 'Ea(%)', 'Et(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x - x_i) / x) * 100
    approximate_relative_error = 1
    errors = {}
    while (true_relative_error > 0.005 and approximate_relative_error > 0.005) or iter_num <= 10:
        values += "%-25s%-25s%-25s%-25s%-25s\n" % (iter_num, x_i, f_x_i, approximate_relative_error if iter_num != 1 else '---', true_relative_error)

        pre_x_i = x_i
        x_i = x_i_1
        x_i_1 = x_i - f_x.subs(y, x_i) / diff(f_x, y, 1).subs(y, x_i)
        f_x_i = f_x.subs(y, x_i)

        iter_num += 1
        true_relative_error = abs((x - x_i) / x) * 100
        approximate_relative_error = abs((x_i - pre_x_i) / x_i) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]
    print(header)
    print(values)

    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for Newton Raphson method')

    plt.show()

    return errors


def secant(x, x_i, x_i_1, f_x):
    print('Secant Method: ')

    # x = 0.119737
    # f_x = 4.51 * y + 1.36 * y**2 - 0.56
    #
    # x_i = 0
    # x_i_1 = 1

    f_x_i = f_x.subs(y, x_i)
    f_x_i_1 = f_x.subs(y, x_i_1)

    x_i_2 = x_i_1 - (f_x.subs(y, x_i_1) * (x_i - x_i_1)) / (f_x.subs(y, x_i) - f_x.subs(y, x_i_1))

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'Xi-1', 'Xi', 'Xi+1', 'f(Xi-1)', 'f(Xi)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x - x_i_2) / x) * 100
    approximate_relative_error = 1
    errors = {}
    while iter_num <= 10:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (iter_num, x_i, x_i_1, x_i_2, f_x_i_1, f_x_i, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        x_i = x_i_1
        x_i_1 = x_i_2
        x_i_2 = x_i_1 - (f_x.subs(y, x_i_1) * (x_i - x_i_1)) / (f_x.subs(y, x_i) - f_x.subs(y, x_i_1))

        f_x_i = f_x.subs(y, x_i)
        f_x_i_1 = f_x.subs(y, x_i_1)

        iter_num += 1
        true_relative_error = abs((x - x_i_2) / x) * 100
        approximate_relative_error = abs((x_i_2 - x_i_1) / x_i_2) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]
    print(header)
    print(values)

    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for Secant method')

    plt.show()

    return errors


x_l, x_u = bracketing_methods_initial_guesses()
bi_section_errors = bi_section(x, x_l, x_u, f_x)
false_position_errors = false_position(x, x_l, x_u, f_x)

x_i, x_i_1 = open_methods_initial_guesses()
fixed_point_errors = fixed_point(x, x_i, f_x)
newton_raphson_errors = newton_raphson(x, x_i, f_x)
secant_errors = secant(x, x_i, x_i_1, f_x)

plt.plot(bi_section_errors.keys(), [error[0] for error in bi_section_errors.values()], label="Et(%) for BiSection")
plt.plot(false_position_errors.keys(), [error[0] for error in false_position_errors.values()], label="Et(%) for False Position")
plt.plot(fixed_point_errors.keys(), [error[0] for error in fixed_point_errors.values()], label="Et(%) for Fixed Point")
plt.plot(newton_raphson_errors.keys(), [error[0] for error in newton_raphson_errors.values()], label="Et(%) for Newton Raphson")
plt.plot(secant_errors.keys(), [error[0] for error in secant_errors.values()], label="Et(%) for Secant")
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.xlabel('iteration')
plt.ylabel('error(%)')
plt.title('True errors for all method')
plt.show()


plt.plot(bi_section_errors.keys(), [error[1] for error in bi_section_errors.values()], label="Ea(%) for BiSection")
plt.plot(false_position_errors.keys(), [error[1] for error in false_position_errors.values()], label="Ea(%) for False Position")
plt.plot(fixed_point_errors.keys(), [error[1] for error in fixed_point_errors.values()], label="Ea(%) for Fixed Point")
plt.plot(newton_raphson_errors.keys(), [error[1] for error in newton_raphson_errors.values()], label="Ea(%) for Newton Raphson")
plt.plot(secant_errors.keys(), [error[1] for error in secant_errors.values()], label="Ea(%) for Secant")
plt.legend()
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.xlabel('iteration')
plt.ylabel('error(%)')
plt.title('Approximate errors for all method')
plt.show()
