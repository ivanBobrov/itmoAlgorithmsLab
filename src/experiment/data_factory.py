import numpy as np

class DataFactory:

    def get_vector(self, size):
        return np.random.rand(size)

    def get_matrix(self, size):
        return np.random.rand(size, size)