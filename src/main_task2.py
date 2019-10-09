import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from task2.exhaustive_search import ExhaustiveSearch
from task2.dichotomy import Dichotomy
from task2.golden_section import GoldenSection

# Part 1


def cubic_function(x):
    return pow(x, 3)


def abs_function(x):
    return abs(x - 0.2)


def sin_function(x):
    return x * math.sin(1/x)


for optimization_method in ([ExhaustiveSearch(), Dichotomy(), GoldenSection()]):
    result = optimization_method.find_minimum(cubic_function, 0, 1, 0.001)
    print(optimization_method.get_name() + " | cubic function: " + str(result.x_minimum) +
          "; iterations: " + str(result.iterations_number) +
          "; function calls: " + str(result.function_calls))

for optimization_method in ([ExhaustiveSearch(), Dichotomy(), GoldenSection()]):
    result = optimization_method.find_minimum(abs_function, 0, 1, 0.001)
    print(optimization_method.get_name() + " | abs function: " + str(result.x_minimum) +
          "; iterations: " + str(result.iterations_number) +
          "; function calls: " + str(result.function_calls))

for optimization_method in ([ExhaustiveSearch(), Dichotomy(), GoldenSection()]):
    result = optimization_method.find_minimum(sin_function, 0.1, 1, 0.001)
    print(optimization_method.get_name() + " | sin function: " + str(result.x_minimum) +
          "; iterations: " + str(result.iterations_number) +
          "; function calls: " + str(result.function_calls))


# Part 2


print("\n\n\n============= Part 2 =============")
random.seed(237)
alpha = random.random()
beta = random.random()
print("alpha: " + str(alpha) + "; beta: " + str(beta))

x_values = np.linspace(0, 1, 101)
y_values = []
for i in range(101):
    y_noise = alpha * x_values[i] + beta + random.gauss(0, 1)
    y_values.append(y_noise)

plt.figure(dpi=300)
plt.plot(x_values, y_values, linestyle='none', marker='o', markersize='1', label='data')


def linear_squares_sum(alpha, beta, x_values, y_values):
    s = 0
    for i in range(101):
        s += (alpha * x_values[i] + beta - y_values[i]) ** 2
    return s


def rational_squares_sum(alpha, beta, x_values, y_values):
    s = 0
    for i in range(101):
        s += ((alpha / (1 + beta * x_values[i])) - y_values[i]) ** 2

    return s


a_exhaustive_result = 0.5
b_exhaustive_result = 0.5
lsq_result = rational_squares_sum(a_exhaustive_result, b_exhaustive_result, x_values, y_values)
for a_i in range(2001):
    for b_j in range(2001):
        a = -1 + a_i * 0.001
        b = -1 + b_j * 0.001
        lsq = rational_squares_sum(a, b, x_values, y_values)
        if lsq < lsq_result:
            a_exhaustive_result = a
            b_exhaustive_result = b
            lsq_result = lsq

#y_exhaustive_values = a_exhaustive_result * x_values + b_exhaustive_result
y_exhaustive_values = a_exhaustive_result / (1 + x_values * b_exhaustive_result)
print("alpha: " + str(a_exhaustive_result) + "; beta: " + str(b_exhaustive_result) + "; lsq: " + str(lsq_result))
plt.plot(x_values, y_exhaustive_values, linestyle='-', label='exhaustive')

a_gauss_result = 0.5
b_gauss_result = 0.5
gauss_epsilon = 0.001
lsq_result = rational_squares_sum(a_gauss_result, b_gauss_result, x_values, y_values)
optimizationMethod = GoldenSection()
while True:
    a_new = optimizationMethod.find_minimum((lambda a : rational_squares_sum(a, b_gauss_result, x_values, y_values)),
                                            -1, 1, 0.001).x_minimum
    b_new = optimization_method.find_minimum((lambda b: rational_squares_sum(a_new, b, x_values, y_values)),
                                             -1, 1, 0.001).x_minimum
    if abs(a_new - a_gauss_result) < gauss_epsilon and abs(b_new - b_gauss_result) < gauss_epsilon:
        a_gauss_result = a_new
        b_gauss_result = b_new
        break
    else:
        a_gauss_result = a_new
        b_gauss_result = b_new

#y_gauss_values = a_gauss_result * x_values + b_gauss_result
y_gauss_values = a_gauss_result / (1 + x_values * b_gauss_result)
print("alpha: " + str(a_gauss_result) + "; beta: " + str(b_gauss_result))
plt.plot(x_values, y_gauss_values, linestyle='--', label='gauss')

nelder_mead_result = minimize(lambda x: rational_squares_sum(x[0], x[1], x_values, y_values),
                              np.array([0.5, 0.5]), method='Nelder-Mead', options={'xatol': 0.001})
print(nelder_mead_result.x)
#y_nelder_values = nelder_mead_result.x[0] * x_values + nelder_mead_result.x[1]
y_nelder_values = nelder_mead_result.x[0] / (1 + x_values * nelder_mead_result.x[1])
plt.plot(x_values, y_nelder_values, linestyle='dotted', label='nelder-mead')


plt.title("Rational regression function")
plt.legend(frameon=False)
plt.show()