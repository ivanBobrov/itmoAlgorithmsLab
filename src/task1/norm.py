import math
from task1.experiment.abstract_experiment import Experiment


class NormExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = 'The Euclidean norm of the elements'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        return math.sqrt(sum(map((lambda x: x * x), vector)))
