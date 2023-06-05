import matplotlib.pyplot as plt
from sympy import diff, symbols, N, sympify, Matrix
import matplotlib

matplotlib.use('TkAgg')

# f_x_y_1_for_numpy = lambdify((x, y), f_x_y_1, modules=['numpy'])
    # f_x_y_2_for_numpy = lambdify((x, y), f_x_y_2, modules=['numpy'])
    #
    # x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
    # y_values = np.linspace(ranges['y'][0], ranges['y'][1], 100)
    #
    # x_values, y_values = np.meshgrid(x_values, y_values)
    #
    # f_x_y_1_values = f_x_y_1_for_numpy(x_values, y_values)
    # f_x_y_2_values = f_x_y_2_for_numpy(x_values, y_values)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # ax.plot_surface(x_values, y_values, f_x_y_1_values)
    # ax.plot_surface(x_values, y_values, f_x_y_2_values)
    #
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_title('f1,2(x, y)')
    #
    # ax.plot([x_exact], [y_exact], [f_x_y_1_exact], marker='o', color='red', markersize=5, label='f1(x, y)')
    # ax.plot([x_exact], [y_exact], [f_x_y_2_exact], marker='*', color='blue', markersize=5, label='f2(x, y)')
    # ax.legend()
    #
    # plt.show()


def draw_approximate_and_true_errors(errors):
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


def non_linear_system_inputs():
    f_1 = input('enter your first function: ')
    f_2 = input('enter your second function: ')

    x_exact = input('enter your exact value of x: ')
    y_exact = input('enter your exact value of y: ')

    x_i = input('enter the initial value of x: ')
    y_i = input('enter the initial value of y: ')

    approximate_relative_error_cond_x = input('enter the approximate error condition value for x: ')
    approximate_relative_error_cond_y = input('enter the approximate error condition value for y: ')
    true_relative_error_cond_x = input('enter the true error condition value for x: ')
    true_relative_error_cond_y = input('enter the true error condition value for y: ')
    iter_num_cond = input('enter the iteration number condition value: ')

    return sympify(f_1), sympify(f_2), float(x_exact), float(y_exact), float(x_i), float(y_i), \
           float(approximate_relative_error_cond_x), float(approximate_relative_error_cond_y), \
           float(true_relative_error_cond_x), float(true_relative_error_cond_y), int(iter_num_cond)


def newton_raphson_for_non_linear_system(f_1, f_2, x_exact, y_exact, x_i, y_i, approximate_relative_error_cond_x=0.005, approximate_relative_error_cond_y=0.005, true_relative_error_cond_x=0.005, true_relative_error_cond_y=0.005, iter_num_cond=10):

    X = Matrix([[x_i], [y_i]])

    F = Matrix([[N(f_1.subs({x: X[0], y: X[1]}))], [N(f_2.subs({x: X[0], y: X[1]}))]])

    df1_dx = diff(f_1, x, 1)
    df1_dy = diff(f_1, y, 1)
    df2_dx = diff(f_2, x, 1)
    df2_dy = diff(f_2, y, 1)

    J_x = Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])

    true_relative_error_x = abs((x_exact - X[0]) / x_exact)
    true_relative_error_y = abs((y_exact - X[1]) / y_exact)

    approximate_relative_error_x = 10
    approximate_relative_error_y = 10

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % (
        'Iteration', 'x', 'y', 'f1(x, y)', 'f2(x, y)', 'Et(%)(x)', 'Et(%)(y)', 'Ea(%)(x)', 'Ea(%)(y)')
    values = ""

    iter_num = 1
    errors = {}
    while (true_relative_error_x > true_relative_error_cond_x or true_relative_error_y > true_relative_error_cond_y) and \
            (approximate_relative_error_x > approximate_relative_error_cond_x or approximate_relative_error_y > approximate_relative_error_cond_y)\
            and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, X[0], X[1], F[0], F[1], true_relative_error_x, true_relative_error_y,
            approximate_relative_error_x if iter_num != 1 else '---',
            approximate_relative_error_y if iter_num != 1 else '---')

        df1_dx = N(J_x[0, 0].subs({x: X[0], y: X[1]}))
        df1_dy = N(J_x[0, 1].subs({x: X[0], y: X[1]}))
        df2_dx = N(J_x[1, 0].subs({x: X[0], y: X[1]}))
        df2_dy = N(J_x[1, 1].subs({x: X[0], y: X[1]}))

        J = Matrix([[df1_dx, df1_dy], [df2_dx, df2_dy]])
        J_inv = J.inv()

        Y = J_inv * (-F)

        pre_x_i = X[0]
        pre_y_i = X[1]

        X = X + Y
        F = Matrix([[N(f_1.subs({x: X[0], y: X[1]}))], [N(f_2.subs({x: X[0], y: X[1]}))]])

        true_relative_error_x = abs((x_exact - X[0]) / x_exact)
        true_relative_error_y = abs((y_exact - X[1]) / y_exact)

        if iter_num == 0:
            iter_num += 1
            continue

        approximate_relative_error_x = abs((X[0] - pre_x_i) / X[0])
        approximate_relative_error_y = abs((X[1] - pre_y_i) / X[1])

        errors[iter_num] = [true_relative_error_x, true_relative_error_y, approximate_relative_error_x,
                            approximate_relative_error_y]
        iter_num += 1

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)


x = symbols('x')
y = symbols('y')

f_1, f_2, x_exact, y_exact, x_i, y_i, approximate_relative_error_cond_x, approximate_relative_error_cond_y, true_relative_error_cond_x, true_relative_error_cond_y, iter_num_cond = non_linear_system_inputs()
newton_raphson_for_non_linear_system(f_1, f_2, x_exact, y_exact, x_i, y_i, approximate_relative_error_cond_x, approximate_relative_error_cond_y, true_relative_error_cond_x, true_relative_error_cond_y, iter_num_cond)
