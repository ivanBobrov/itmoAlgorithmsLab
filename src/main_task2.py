import math
from task2.exhaustive_search import ExhaustiveSearch
from task2.dichotomy import Dichotomy
from task2.golden_section import GoldenSection


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
