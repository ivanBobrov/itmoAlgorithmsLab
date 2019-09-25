# Main module to run all experiments
import matplotlib.pyplot as plt
import numpy as np
from task1.constant import ConstantExperiment
from task1.sum import SumExperiment
from task1.product import ProductExperiment
from task1.norm import NormExperiment
from task1.polynomial_direct import PolynomialDirectExperiment
from task1.polynomial_horner import PolynomialHornerExperiment
from task1.bubble_sort import BubbleSortExperiment
from task1.matrix_product import MatrixProductExperiment

full_experiment_list = [ConstantExperiment(), SumExperiment(), ProductExperiment(), NormExperiment(),
                        PolynomialDirectExperiment(), PolynomialHornerExperiment(), BubbleSortExperiment(),
                        MatrixProductExperiment()]

experiment_list = [MatrixProductExperiment()]

for experiment in experiment_list:
    print("\nstarting: " + experiment.get_name())
    x = []
    y = []

    #singleTime = experiment.start(1)
    #x.append(1)
    #y.append(1)

    for size in range(50, 505, 50):
        averageTime = experiment.start(size)
        print("Normalized time for size '" + str(size) + "' is: " + str(averageTime))

        x.append(size)
        y.append(averageTime)

    plt.figure(dpi=300)
    poly_theoretical = np.polyfit(x, y, 3)
    x_poly = np.linspace(x[0], x[-1], 50)
    y_poly = np.polyval(poly_theoretical, x_poly)

    plt.plot(x, y, linestyle='none', marker='o', markersize='1', label='actual')
    plt.plot(x_poly, y_poly, label='theoretical')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(experiment.get_name())
    plt.xlabel('Size, n')
    plt.ylabel('Running time')
    plt.legend(frameon=False)
    plt.savefig(fname=experiment.get_name() + '.png', format='png')
