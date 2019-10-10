from .optimization_method import OptimizationMethod
from .optimization_method import OptimizationResult


class Dichotomy(OptimizationMethod):

    def get_name(self):
        return "Dichotomy"

    def find_minimum(self, function, a_from, b_to, epsilon):
        iterations_number = 0
        function_calls = 0

        x_left = a_from
        x_right = b_to
        delta = epsilon * 0.01

        while x_right - x_left > epsilon:
            x_x1 = (x_left + x_right - delta) / 2
            x_x2 = (x_left + x_right + delta) / 2

            if function(x_x1) >= function(x_x2):
                x_left = x_x1
            else:
                x_right = x_x2

            iterations_number += 1
            function_calls += 2

        x_minimum = (x_left + x_right) / 2
        return OptimizationResult(x_minimum, iterations_number, function_calls)