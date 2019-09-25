from experiment.abstract_experiment import Experiment


class BubbleSortExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = 'Bubble sort'

    def get_data(self, size, factory):
        return factory.get_vector(size)

    def run(self, vector):
        changed = True
        while changed:
            changed = False
            for i in range(0, len(vector) - 1):
                if vector[i] > vector[i + 1]:
                    tmp = vector[i]
                    vector[i] = vector[i+1]
                    vector[i+1] = tmp
                    changed = True

        return vector
