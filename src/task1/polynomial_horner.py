from experiment.abstract_experiment import Experiment


class PolynomialHornerExperiment(Experiment):
    POLYNOMIAL_POINT_X = 1.5

    def __init__(self):
        super().__init__()
        self.experimentName = 'Polynomial Horner\'s calculation for x = 1.5'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        sum = 0
        for i in range(len(vector) - 1, -1, -1):
            sum = sum * self.POLYNOMIAL_POINT_X + vector[i]

        return sum
