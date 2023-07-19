from sympy import diff, symbols, sympify
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# f_x_for_numpy = lambdify(y, f_x, modules=['numpy'])
# x_values = np.linspace(ranges['x'][0], ranges['x'][1], 100)
# y_values = f_x_for_numpy(x_values)
#
# plt.plot(x_values, y_values)
# plt.plot(x, f_x_exact, 'ro', label='Data Point')
# plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
# plt.text(x, f_x_exact, f'({x}, {f_x_exact})', verticalalignment='bottom', horizontalalignment='left')
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.title('My Function')
#
# plt.show()


def draw_approximate_and_true_errors(errors):
    plt.plot(errors.keys(), [error[0] for error in errors.values()], label="Et(%)")
    plt.plot(errors.keys(), [error[1] for error in errors.values()], label="Ea(%)")
    plt.legend()

    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
    plt.xlabel('iteration')
    plt.ylabel('error(%)')

    plt.title('True and approximate errors for BiSection method')

    plt.show()


def bracketing_methods_inputs():
    f_x = input('enter your function: ')
    x_exact = input('enter your exact solution: ')
    x_lower = input('enter the lower bracket: ')
    x_upper = input('enter the upper bracket: ')
    approximate_relative_error_cond = input('enter the approximate error condition value: ')
    true_relative_error_cond = input('enter the true error condition value: ')
    iter_num_cond = input('enter the iteration number condition value: ')
    return sympify(f_x), float(x_exact), float(x_lower), float(x_upper), float(approximate_relative_error_cond), float(
        true_relative_error_cond), int(iter_num_cond)


def bi_section(f_x, x_exact, x_lower, x_upper, approximate_relative_error_cond=0.005, true_relative_error_cond=0.005,
               iter_num_cond=10):
    print('BiSection Method: ')
    f_x_l = f_x.subs(x, x_lower)
    f_x_u = f_x.subs(x, x_upper)

    if f_x_u * f_x_l > 0:
        exit('lower and upper not bracketing the exact solution')

    x_root = (x_upper + x_lower) / 2
    f_x_r = f_x.subs(x, x_root)

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % (
    'Iteration', 'x lower', 'x upper', 'x root', 'f(xr)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x_exact - x_root) / x_exact) * 100
    approximate_relative_error = 1
    errors = {}
    while true_relative_error > true_relative_error_cond and approximate_relative_error > approximate_relative_error_cond and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_lower, x_upper, x_root, f_x_r, true_relative_error,
            approximate_relative_error if iter_num != 1 else '---')

        if f_x_r * f_x_u < 0:
            x_lower = x_root
            f_x_l = f_x_r
        else:
            x_upper = x_root
            f_x_u = f_x_r

        pre_x_r = x_root
        x_root = (x_upper + x_lower) / 2
        f_x_r = f_x.subs(x, x_root)

        iter_num += 1
        true_relative_error = abs((x_exact - x_root) / x_exact) * 100
        approximate_relative_error = abs((x_root - pre_x_r) / x_root) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)

    return errors


def false_position(f_x, x_exact, x_lower, x_upper, approximate_relative_error_cond=0.005,
                   true_relative_error_cond=0.005, iter_num_cond=10):
    print('False Position Method: ')
    f_x_l = f_x.subs(x, x_lower)
    f_x_u = f_x.subs(x, x_upper)

    if f_x_u * f_x_l > 0:
        exit('lower and upper not bracketing the exact solution')

    x_root = x_upper - f_x_u * (x_upper - x_lower) / (f_x_u - f_x_l)
    f_x_r = f_x.subs(x, x_root)

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % (
    'Iteration', 'x lower', 'x upper', 'x root', 'f(xr)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x_exact - x_root) / x_exact) * 100
    approximate_relative_error = 1
    errors = {}
    while true_relative_error > true_relative_error_cond and approximate_relative_error > approximate_relative_error_cond and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_lower, x_upper, x_root, f_x_r, true_relative_error,
            approximate_relative_error if iter_num != 1 else '---')

        if f_x_r * f_x_u < 0:
            x_lower = x_root
            f_x_l = f_x_r
        else:
            x_upper = x_root
            f_x_u = f_x_r

        pre_x_r = x_root
        x_root = x_upper - f_x_u * (x_upper - x_lower) / (f_x_u - f_x_l)
        f_x_r = f_x.subs(x, x_root)

        iter_num += 1
        true_relative_error = abs((x_exact - x_root) / x_exact) * 100
        approximate_relative_error = abs((x_root - pre_x_r) / x_root) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)

    return errors


def open_methods_inputs():
    f_x = input('enter your function: ')
    g_x = input('enter your g function: ')
    x_exact = input('enter your exact solution: ')
    x_i = input('enter the first initial value: ')
    x_i_1 = input('enter the second initial value: ')
    approximate_relative_error_cond = input('enter the approximate error condition value: ')
    true_relative_error_cond = input('enter the true error condition value: ')
    iter_num_cond = input('enter the iteration number condition value: ')
    return sympify(f_x), sympify(g_x), float(x_exact), float(x_i), float(x_i_1), float(
        approximate_relative_error_cond), float(true_relative_error_cond), int(iter_num_cond)


def fixed_point(f_x, g_x, x_exact, x_i, approximate_relative_error_cond=0.005, true_relative_error_cond=0.005,
                iter_num_cond=10):
    print('Fixed Point Method: ')

    f_x_i = f_x.subs(x, x_i)
    g_x_i = g_x.subs(x, x_i)

    g_x_d_x = diff(g_x, x, 1)
    k = abs(g_x_d_x.subs(x, x_exact))
    if k > 1:
        exit('almost this g(x) will diverge because |k| > 1')

    header = "%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'x', 'f(x)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x_exact - x_i) / x_exact) * 100
    approximate_relative_error = 1
    errors = {}
    while true_relative_error > true_relative_error_cond and approximate_relative_error > approximate_relative_error_cond and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_i, f_x_i, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        pre_x_i = x_i

        x_i = g_x_i
        g_x_i = g_x.subs(x, x_i)
        f_x_i = f_x.subs(x, x_i)

        iter_num += 1
        true_relative_error = abs((x_exact - x_i) / x_exact) * 100
        approximate_relative_error = abs((x_i - pre_x_i) / x_i) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)

    return errors


def newton_raphson(f_x, x_exact, x_i, approximate_relative_error_cond=0.005, true_relative_error_cond=0.005,
                   iter_num_cond=10):
    print('Newton Raphson Method: ')

    f_x_i = f_x.subs(x, x_i)
    x_i_1 = x_i - f_x.subs(x, x_i) / diff(f_x, x, 1).subs(x, x_i)

    header = "%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'x', 'f(x)', 'Ea(%)', 'Et(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x_exact - x_i) / x_exact) * 100
    approximate_relative_error = 1
    errors = {}
    while true_relative_error > true_relative_error_cond and approximate_relative_error > approximate_relative_error_cond and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s\n" % (
            iter_num, x_i, f_x_i, approximate_relative_error if iter_num != 1 else '---', true_relative_error)

        pre_x_i = x_i

        x_i = x_i_1
        f_x_i = f_x.subs(x, x_i)
        x_i_1 = x_i - f_x.subs(x, x_i) / diff(f_x, x, 1).subs(x, x_i)

        iter_num += 1
        true_relative_error = abs((x_exact - x_i) / x_exact) * 100
        approximate_relative_error = abs((x_i - pre_x_i) / x_i) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)

    return errors


def secant(f_x, x_exact, x_i, x_i_1, approximate_relative_error_cond=0.005, true_relative_error_cond=0.005,
                   iter_num_cond=10):
    print('Secant Method: ')

    f_x_i = f_x.subs(x, x_i)
    f_x_i_1 = f_x.subs(x, x_i_1)

    x_i_2 = x_i_1 - (f_x.subs(x, x_i_1) * (x_i - x_i_1)) / (f_x.subs(x, x_i) - f_x.subs(x, x_i_1))

    header = "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s" % ('Iteration', 'Xi-1', 'Xi', 'Xi+1', 'f(Xi-1)', 'f(Xi)', 'Et(%)', 'Ea(%)')
    values = ""

    iter_num = 1
    true_relative_error = abs((x_exact - x_i_2) / x_exact) * 100
    approximate_relative_error = 1
    errors = {}
    while true_relative_error > true_relative_error_cond and approximate_relative_error > approximate_relative_error_cond and iter_num <= iter_num_cond:
        values += "%-25s%-25s%-25s%-25s%-25s%-25s%-25s%-25s\n" % (iter_num, x_i, x_i_1, x_i_2, f_x_i_1, f_x_i, true_relative_error, approximate_relative_error if iter_num != 1 else '---')

        x_i = x_i_1
        x_i_1 = x_i_2
        x_i_2 = x_i_1 - (f_x.subs(x, x_i_1) * (x_i - x_i_1)) / (f_x.subs(x, x_i) - f_x.subs(x, x_i_1))

        f_x_i = f_x.subs(x, x_i)
        f_x_i_1 = f_x.subs(x, x_i_1)

        iter_num += 1
        true_relative_error = abs((x_exact - x_i_2) / x_exact) * 100
        approximate_relative_error = abs((x_i_2 - x_i_1) / x_i_2) * 100

        errors[iter_num] = [true_relative_error, approximate_relative_error]

    print(header)
    print(values)

    draw_approximate_and_true_errors(errors)

    return errors


x = symbols("x")

f_x, x_exact, x_lower, x_upper, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond = bracketing_methods_inputs()
bi_section_errors = bi_section(f_x, x_exact, x_lower, x_upper, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond)
false_position_errors = false_position(f_x, x_exact, x_lower, x_upper, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond)

f_x, g_x, x_exact, x_i, x_i_1, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond = open_methods_inputs()
fixed_point_errors = fixed_point(f_x, g_x, x_exact, x_i, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond)
newton_raphson_errors = newton_raphson(f_x, x_exact, x_i, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond)
secant_errors = secant(f_x, x_exact, x_i, x_i_1, approximate_relative_error_cond, true_relative_error_cond, iter_num_cond)


# plt.plot(bi_section_errors.keys(), [error[0] for error in bi_section_errors.values()], label="Et(%) for BiSection")
# plt.plot(false_position_errors.keys(), [error[0] for error in false_position_errors.values()], label="Et(%) for False Position")
# plt.plot(fixed_point_errors.keys(), [error[0] for error in fixed_point_errors.values()], label="Et(%) for Fixed Point")
# plt.plot(newton_raphson_errors.keys(), [error[0] for error in newton_raphson_errors.values()], label="Et(%) for Newton Raphson")
# plt.plot(secant_errors.keys(), [error[0] for error in secant_errors.values()], label="Et(%) for Secant")
# plt.legend()
# plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
# plt.xlabel('iteration')
# plt.ylabel('error(%)')
# plt.title('True errors for all method')
# plt.show()
#
#
# plt.plot(bi_section_errors.keys(), [error[1] for error in bi_section_errors.values()], label="Ea(%) for BiSection")
# plt.plot(false_position_errors.keys(), [error[1] for error in false_position_errors.values()], label="Ea(%) for False Position")
# plt.plot(fixed_point_errors.keys(), [error[1] for error in fixed_point_errors.values()], label="Ea(%) for Fixed Point")
# plt.plot(newton_raphson_errors.keys(), [error[1] for error in newton_raphson_errors.values()], label="Ea(%) for Newton Raphson")
# plt.plot(secant_errors.keys(), [error[1] for error in secant_errors.values()], label="Ea(%) for Secant")
# plt.legend()
# plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
# plt.xlabel('iteration')
# plt.ylabel('error(%)')
# plt.title('Approximate errors for all method')
# plt.show()
