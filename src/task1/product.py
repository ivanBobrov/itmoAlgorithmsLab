import functools
from experiment.abstract_experiment import Experiment


class ProductExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = 'The product of elements'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        return functools.reduce((lambda x, y: x * y), vector)