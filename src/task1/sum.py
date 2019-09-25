from experiment.abstract_experiment import Experiment


class SumExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = 'The sum of elements'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        return sum(vector)