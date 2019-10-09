import math
from .optimization_method import OptimizationMethod
from .optimization_method import OptimizationResult


class GoldenSection(OptimizationMethod):
    GOLDEN_RATIO = (3 - math.sqrt(5)) / 2

    def get_name(self):
        return "Golden section"

    def find_minimum(self, function, a_from, b_to, epsilon):
        iterations_number = 0
        function_calls = 2

        x_left = a_from
        x_right = b_to

        x_x1 = x_left + (x_right - x_left) * self.GOLDEN_RATIO
        x_x2 = x_right - (x_right - x_left) * self.GOLDEN_RATIO
        f_x1 = function(x_x1)
        f_x2 = function(x_x2)

        while x_right - x_left > epsilon:
            if f_x1 >= f_x2:
                x_left = x_x1

                x_x1 = x_x2
                f_x1 = f_x2

                x_x2 = x_right - (x_right - x_left) * self.GOLDEN_RATIO
                f_x2 = function(x_x2)
            else:
                x_right = x_x2

                x_x2 = x_x1
                f_x2 = f_x1

                x_x1 = x_left + (x_right - x_left) * self.GOLDEN_RATIO
                f_x1 = function(x_x1)

            iterations_number += 1
            function_calls += 1

        x_minimum = (x_left + x_right) / 2
        return OptimizationResult(x_minimum, iterations_number, function_calls)
