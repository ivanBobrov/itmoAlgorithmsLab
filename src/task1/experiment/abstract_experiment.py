import timeit
from .data_factory import DataFactory

class Experiment:
    NUMBER_OF_RUNS = 10_000

    def __init__(self):
        self.experimentName = "unspecified"
        self.dataFactory = DataFactory()

    def get_name(self):
        return self.experimentName

    def start(self, size):
        data = self.get_data(size, self.dataFactory)
        timer = timeit.Timer(lambda: self.run(data))
        average_time = timer.timeit(self.NUMBER_OF_RUNS)
        return average_time / self.NUMBER_OF_RUNS

    def get_data(self, size, factory):
        pass

    def run(self, data):
        pass
