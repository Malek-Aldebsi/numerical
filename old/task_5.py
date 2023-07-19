import numpy as np
import sympy
import matplotlib.pyplot as plt
import math
import random
from sympy import diff, symbols, factorial, exp, pi, cos, lambdify, shape, N
import matplotlib
matplotlib.use('TkAgg')


def get_random_value(lower_bound, upper_bound):
    random_number = random.uniform(lower_bound, upper_bound)
    if random_number == 0:
        random_number = get_random_value()
    return float(format(random_number, ".3g"))


ranges = {
    'a7.1': [-10, 10],
    'a8.1': [-2, 2],
    'a9.1': [-1, 1],

    'a16.1': [-10, 10],
    'a17.1': [-2, 2],
    'a18.1': [-1, 1],

    'a25.1': [-1, 1],
    'a26.1': [-1, 1],
    'a27.1': [-1, 1],

    'a7.2': [-10, 10],
    'a8.2': [-2, 2],
    'a9.2': [-1, 1],

    'a16.2': [-10, 10],
    'a17.2': [-2, 2],
    'a18.2': [-1, 1],

    'a25.2': [-1, 1],
    'a26.2': [-1, 1],
    'a27.2': [-1, 1],

    'x': [-2, 2],
    'y': [-2, 2]
}

coefficientsValues = {}
for coff, coff_range in ranges.items():
    random_num = get_random_value(coff_range[0], coff_range[1])
    coefficientsValues[coff] = random_num

x_exact = coefficientsValues.pop('x')
y_exact = coefficientsValues.pop('y')

y = symbols('y')
x = symbols('x')

f_x_y_1 = coefficientsValues["a7.1"] * x + coefficientsValues["a8.1"] * x**2 + coefficientsValues["a9.1"] * x**3 + coefficientsValues["a16.1"] * y + coefficientsValues["a17.1"] * y**2 + coefficientsValues["a18.1"] * y**3 + coefficientsValues["a25.1"] * x * y + coefficientsValues["a26.1"] * x**2 * y + coefficientsValues["a27.1"] * x**3 * y**2
f_x_y_2 = coefficientsValues["a7.2"] * x + coefficientsValues["a8.2"] * x**2 + coefficientsValues["a9.2"] * x**3 + coefficientsValues["a16.2"] * y + coefficientsValues["a17.2"] * y**2 + coefficientsValues["a18.2"] * y**3 + coefficientsValues["a25.2"] * x * y + coefficientsValues["a26.2"] * x**2 * y + coefficientsValues["a27.2"] * x**3 * y**2


f_x_y_1_exact = f_x_y_1.subs({x: x_exact, y: y_exact})
f_x_y_2_exact = f_x_y_2.subs({x: x_exact, y: y_exact})

f_x_y_1 = f_x_y_1 - f_x_y_1_exact
f_x_y_2 = f_x_y_2 - f_x_y_2_exact
# f_x_y_1 = x ** 2 + y - x - 0.4
# f_x_y_2 = 2 * y + 3 * x * y - 2 * x**3

header = ''
str_coff = ''
for coff in coefficientsValues.keys():
    header += "%-15s" % f'{coff}'
    str_coff += "%-15s" % (coefficientsValues[f'{coff}'])

header += "%-15s%-15s%-15s%-15s" % ('x', 'y', 'f1(x, y)', 'f2(x, y)')
str_coff += "%-15s%-15s%-15s%-15s" % (x_exact, y_exact, f_x_y_1_exact, f_x_y_2_exact)

print(header)
print(str_coff)
print('-' * 170)

f_x_y_1_for_numpy = lambdify((x, y), f_x_y_1, modules=['numpy'])
f_x_y_2_for_numpy = lambdify((x, y), f_x_y_2, modules=['numpy'])

x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
y_values = np.linspace(ranges['y'][0], ranges['y'][1], 100)

x_values, y_values = np.meshgrid(x_values, y_values)

f_x_y_1_values = f_x_y_1_for_numpy(x_values, y_values)
f_x_y_2_values = f_x_y_2_for_numpy(x_values, y_values)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x_values, y_values, f_x_y_1_values)
ax.plot_surface(x_values, y_values, f_x_y_2_values)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('f1,2(x, y)')

ax.plot([x_exact], [y_exact], [f_x_y_1_exact], marker='o', color='red', markersize=5, label='f1(x, y)')
ax.plot([x_exact], [y_exact], [f_x_y_2_exact], marker='*', color='blue', markersize=5, label='f2(x, y)')
ax.legend()

plt.show()

x_i = get_random_value(ranges['x'][0], ranges['x'][1])
y_i = get_random_value(ranges['y'][0], ranges['y'][1])

X = sympy.Matrix([[x_i], [y_i]])
# X = sympy.Matrix([[-5], [0]])

F = sympy.Matrix([[N(f_x_y_1.subs({x: X[0], y: X[1]}))], [N(f_x_y_2.subs({x: X[0], y: X[1]}))]])

df1_dx = diff(f_x_y_1, x, 1)
df1_dy = diff(f_x_y_1, y, 1)
df2_dx = diff(f_x_y_2, x, 1)
df2_dy = diff(f_x_y_2, y, 1)

J_x = sympy.Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])
print(J_x)

true_relative_error_x = abs((x_exact - X[0]) / x_exact)
true_relative_error_y = abs((y_exact - X[1]) / y_exact)

approximate_relative_error_x = 10
approximate_relative_error_y = 10

header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'x', 'y', 'f1(x, y)', 'f2(x, y)', 'Et(%)(x)', 'Et(%)(y)', 'Ea(%)(x)', 'Ea(%)(y)')
values = ""
iter_num = 1
errors = {}
while ((true_relative_error_x > 0.005 or true_relative_error_y > 0.005) and (approximate_relative_error_x > 0.005 or approximate_relative_error_y > 0.005)) or iter_num <= 4:
    values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (iter_num, X[0], X[1], F[0], F[1], true_relative_error_x, true_relative_error_y, approximate_relative_error_x if iter_num != 0 else '---', approximate_relative_error_y if iter_num != 0 else '---')

    df1_dx = N(J_x[0, 0].subs({x: X[0], y: X[1]}))
    df1_dy = N(J_x[0, 1].subs({x: X[0], y: X[1]}))
    df2_dx = N(J_x[1, 0].subs({x: X[0], y: X[1]}))
    df2_dy = N(J_x[1, 1].subs({x: X[0], y: X[1]}))

    J = sympy.Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])
    J_inv = J.inv()

    Y = J_inv * (-F)
    print(f'\niteration num {iter_num}')
    print(f'X({iter_num})= ')
    print(X)
    print(f'J(X({iter_num}))= ')
    print(J)
    print(f'F(X({iter_num}))= ')
    print(F)
    print(f'Y({iter_num})= ')
    print(Y)

    pre_x_i = X[0]
    pre_y_i = X[1]

    X = X + Y
    F = sympy.Matrix([[N(f_x_y_1.subs({x: X[0], y: X[1]}))], [N(f_x_y_2.subs({x: X[0], y: X[1]}))]])

    print(f"Absolute Error x= {abs(true_relative_error_x * x_exact)}")
    print(f"Absolute Error y= {abs(true_relative_error_y * y_exact)}")

    print(f"True Relative Error x= {true_relative_error_x}")
    print(f"True Relative Error y= {true_relative_error_y}")

    true_relative_error_x = abs((x_exact - X[0]) / x_exact)
    true_relative_error_y = abs((y_exact - X[1]) / y_exact)

    if iter_num == 0:
        print("Approximate Relative Error x= ---")
        print("Approximate Relative Error y= ---")
        iter_num += 1
        continue

    approximate_relative_error_x = abs((X[0] - pre_x_i) / X[0])
    approximate_relative_error_y = abs((X[1] - pre_y_i) / X[1])

    print(f"Approximate Relative Error x= {approximate_relative_error_x}")
    print(f"Approximate Relative Error y= {approximate_relative_error_y}")

    errors[iter_num] = [true_relative_error_x, true_relative_error_y, approximate_relative_error_x, approximate_relative_error_y]
    iter_num += 1

print(header)
print(values)

plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)(x)")
plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Et(%)(y)")
plt.plot(errors.keys(), [error[2] for error in errors.values()], label="Ea(%)(x)")
plt.plot(errors.keys(), [error[3] for error in errors.values()], label="Ea(%)(y)")
plt.legend()

plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.xlabel('iteration')
plt.ylabel('error(%)')
plt.title('True and approximate errors')

plt.show()
