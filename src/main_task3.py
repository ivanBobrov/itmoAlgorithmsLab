import random
import numpy as np
import matplotlib.pyplot as plt
from task2.golden_section import GoldenSection
from scipy.optimize import least_squares

# Generating random values for the noisy data
random.seed(86758)
alpha = random.random()
beta = random.random()
print("alpha: " + str(alpha) + "; beta: " + str(beta) + "\n")

# Building up noisy data
x = np.linspace(0, 1, 1001)
y = []
for i in range(1001):
    y_with_noise = alpha * x[i] + beta + random.gauss(0, 1)
    y.append(y_with_noise)


def linear_function(s):
    return s[0] * x + s[1]


def linear_minimization_function(alpha, beta):
    s = 0
    for i in range(len(x)):
        s += (alpha * x[i] + beta - y[i]) ** 2

    return s


def rational_function(s):
    return s[0] / (1 + s[1] * x)


def rational_minimization_function(alpha, beta):
    s = 0
    for i in range(len(x)):
        s += ((alpha / (1 + beta * x[i])) - y[i]) ** 2

    return s


def obtain_gradient(f, a, b):
    eps = 1e-6
    d_a = (f(a + eps, b) - f(a - eps, b)) / (2 * eps)
    d_b = (f(a, b + eps) - f(a, b - eps)) / (2 * eps)
    return d_a, d_b


def obtain_hessian(f, a, b):
    eps = 1e-6
    d_a_a = (f(a + eps, b) - 2 * f(a, b) + f(a - eps, b)) / (eps ** 2)
    d_b_b = (f(a, b + eps) - 2 * f(a, b) + f(a, b - eps)) / (eps ** 2)
    d_a_b = d_b_a = (f(a + eps, b + eps)
                     - f(a - eps, b + eps)
                     - f(a + eps, b - eps)
                     + f(a - eps, b - eps)) / (4 * eps ** 2)
    return d_a_a, d_a_b, d_b_a, d_b_b


# Gradient descent
def gradient_descent(minimization_function, precision, gamma):
    alpha = 0.5
    beta = 0.5
    iterations = 0

    while True:
        gradient = obtain_gradient(minimization_function, alpha, beta)
        alpha_new = alpha - gamma * gradient[0]
        beta_new = beta - gamma * gradient[1]
        iterations += 1
        if abs(minimization_function(alpha, beta) - minimization_function(alpha_new, beta_new)) < precision:
            return alpha_new, beta_new, iterations
        else:
            alpha = alpha_new
            beta = beta_new


# Conjugate gradient descent
def conjugate_gradient_descent(minimization_function, precision):
    alpha = 0.5
    beta = 0.5
    optimization_method = GoldenSection()
    iterations = 0

    while True:
        gradient = obtain_gradient(minimization_function, alpha, beta)

        def step_minimization_function(x):
            return minimization_function(alpha - x * gradient[0],
                                         beta - x * gradient[1])

        gamma = optimization_method.find_minimum(function=step_minimization_function,
                                                 a_from=0, b_to=0.01, epsilon=precision).x_minimum
        alpha_new = alpha - gamma * gradient[0]
        beta_new = beta - gamma * gradient[1]
        iterations += 1
        if abs(minimization_function(alpha, beta) - minimization_function(alpha_new, beta_new)) < precision:
            return alpha_new, beta_new, iterations
        else:
            alpha = alpha_new
            beta = beta_new


# Newton's method
def newton_method(minimization_function, gamma, precision):
    alpha = 0.8
    beta = -0.3
    iterations = 0

    def get_inverse_matrix(m):
        c = m[1] * m[2] - m[0] * m[3]
        return -m[3] / c, m[1] / c, m[2] / c, -m[0] / c

    while True:
        gradient = obtain_gradient(minimization_function, alpha, beta)
        hessian = obtain_hessian(minimization_function, alpha, beta)
        inverse_hessian = get_inverse_matrix(hessian)

        alpha_new = alpha - gamma * (inverse_hessian[0] * gradient[0] + inverse_hessian[1] * gradient[1])
        beta_new = beta - gamma * (inverse_hessian[2] * gradient[0] + inverse_hessian[3] * gradient[1])
        iterations += 1
        if abs(minimization_function(alpha, beta) - minimization_function(alpha_new, beta_new)) < precision:
            return alpha_new, beta_new, iterations
        else:
            alpha = alpha_new
            beta = beta_new


def run_experiment(minimization_function, test_function, plot_name):
    gradient_descent_result = gradient_descent(minimization_function, precision=0.001, gamma=1e-4)
    print("Gradient descent {" +
          "alpha: " + str(gradient_descent_result[0]) + "; " +
          "beta: " + str(gradient_descent_result[1]) + "} " +
          "obtained with " + str(gradient_descent_result[2]) + " iterations")

    conjugate_gradient_result = conjugate_gradient_descent(minimization_function, precision=0.0001)
    print("Conjugate gradient descent {" +
          "alpha: " + str(conjugate_gradient_result[0]) + "; " +
          "beta: " + str(conjugate_gradient_result[1]) + "} " +
          "obtained with " + str(conjugate_gradient_result[2]) + " iterations")

    newton_result = newton_method(minimization_function, precision=1e-4, gamma=0.1)
    print("Newton's method {" +
          "alpha: " + str(newton_result[0]) + "; " +
          "beta: " + str(newton_result[1]) + "} " +
          "obtained with " + str(newton_result[2]) + " iterations")

    lm_result = least_squares(lambda p: test_function(p) - y, (0.5, 0.5), method='lm')
    print("Levenberg-Marquardt {" +
          "alpha: " + str(lm_result.x[0]) + "; " +
          "beta: " + str(lm_result.x[1]) + "} " +
          "obtained with " + str(lm_result.nfev) + " iterations")

    # Data plots
    plt.figure(dpi=300)
    plt.plot(x, y, linestyle='none', marker='o', markersize='1', label='data')
    plt.plot(x, test_function(gradient_descent_result), label='Gradient descent')
    plt.plot(x, test_function(conjugate_gradient_result), label="Conjugate gradient")
    plt.plot(x, test_function(newton_result), label="Newton's method")
    plt.plot(x, test_function(lm_result.x), label='Levenberg-Marquardt')
    plt.legend(frameon=False)
    plt.title(plot_name)
    plt.show()


print("Linear regression")
run_experiment(linear_minimization_function, linear_function, 'Linear regression')

print("\nRational regression")
run_experiment(rational_minimization_function, rational_function, 'Rational regression')