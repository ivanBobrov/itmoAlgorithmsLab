from .optimization_method import OptimizationMethod
from .optimization_method import OptimizationResult


class ExhaustiveSearch(OptimizationMethod):

    def get_name(self):
        return "Exhaustive search"

    def find_minimum(self, function, a_from, b_to, epsilon):
        x_min = a_from
        y_min = function(a_from)
        iter_count = 0
        func_calls = 1

        x_step = a_from
        while x_step <= b_to:
            y_step = function(x_step)
            if y_step < y_min:
                x_min = x_step
                y_min = y_step

            x_step += epsilon
            iter_count += 1
            func_calls += 1

        return OptimizationResult(x_min, iter_count, func_calls)
