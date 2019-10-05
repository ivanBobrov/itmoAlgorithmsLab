from dataclasses import dataclass


class OptimizationMethod:
    def find_minimum(self, function, a_from, b_to, epsilon):
        pass

    def get_name(self):
        return "undefined"


@dataclass
class OptimizationResult:
    x_minimum: float
    iterations_number: int
    function_calls: int
