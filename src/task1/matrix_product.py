from experiment.abstract_experiment import Experiment


class MatrixProductExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.experimentName = 'Matrix product'

    def get_data(self, size, factory):
        return factory.get_matrix(size), factory.get_matrix(size)

    def run(self, matrices):
        A = matrices[0]
        B = matrices[1]

        C = [[0] * len(A) for i in range(len(A))]
        for i in range(len(A[0])):
            for j in range(len(B)):
                for k in range(len(B)):
                    C[i][j] += A[i][k] * B[k][j]
        return C

