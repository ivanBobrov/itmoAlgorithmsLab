from task1.experiment.abstract_experiment import Experiment


class PolynomialDirectExperiment(Experiment):
    POLYNOMIAL_POINT_X = 1.5

    def __init__(self):
        super().__init__()
        self.experimentName = 'Polynomial direct calculation for x = 1.5'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        sum = 0
        for i in range(len(vector)):
            sum += vector[i] * pow(self.POLYNOMIAL_POINT_X, i)

        return sum
