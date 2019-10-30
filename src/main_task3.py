import random
import numpy as np
import matplotlib.pyplot as plt
from task2.golden_section import GoldenSection

# Generating random values for the noisy data
random.seed(86756)
alpha = random.random()
beta = random.random()
print("alpha: " + str(alpha) + "; beta: " + str(beta) + "\n")

# Building up noisy data
x = np.linspace(0, 1, 1001)
y = []
for i in range(1001):
    y_with_noise = alpha * x[i] + beta + random.gauss(0, 0.0001)
    y.append(y_with_noise)

# Initial data plot
plt.figure(dpi=300)
plt.plot(x, y)


def minimization_function(alpha, beta):
    s = 0
    for i in range(len(x)):
        s += (alpha * x[i] + beta - y[i]) ** 2

    return s


def obtain_gradient(minimization_function, a, b):
    epsilon = 1e-6
    a_d = (minimization_function(a + epsilon, b) - minimization_function(a, b)) / epsilon
    b_d = (minimization_function(a, b + epsilon) - minimization_function(a, b)) / epsilon
    return a_d, b_d


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
                                                 a_from=0, b_to=1, epsilon=precision).x_minimum
        alpha_new = alpha - gamma * gradient[0]
        beta_new = beta - gamma * gradient[1]
        iterations += 1
        if abs(minimization_function(alpha, beta) - minimization_function(alpha_new, beta_new)) < precision:
            return alpha_new, beta_new, iterations
        else:
            alpha = alpha_new
            beta = beta_new


gradient_descent_result = gradient_descent(minimization_function, precision=0.00001, gamma=1e-4)
print("Gradient descent {" +
      "alpha: " + str(gradient_descent_result[0]) + "; " +
      "beta: " + str(gradient_descent_result[1]) + "} " +
      "obtained with " + str(gradient_descent_result[2]) + " iterations")

conjugate_gradient_result = conjugate_gradient_descent(minimization_function, precision=0.00001)
print("Conjugate gradient descent {" +
      "alpha: " + str(conjugate_gradient_result[0]) + "; " +
      "beta: " + str(conjugate_gradient_result[1]) + "} " +
      "obtained with " + str(conjugate_gradient_result[2]) + " iterations")
