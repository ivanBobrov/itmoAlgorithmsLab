from task1.experiment.abstract_experiment import Experiment


class ConstantExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = "Constant function"

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        return len(vector)