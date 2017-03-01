import sys, os

#sys.path.append(os.path.dirname(__file__))
#print(sys.path)
os.chdir(os.path.dirname(__file__))

import plot_minimax_survey
import fluent_data_loader as fdl
from matplotlib import pyplot as plt
import numpy as np


def infer_total_utility(fluent_vector):
    loader = fdl.DataLoader()
    normalized_fluent_vector = loader.normalize(fluent_vector)
    utility_functions = plot_minimax_survey.get_utility_functions()

    draw_plot = False

    if draw_plot:
        plt.figure()

    total_utility = 0.

    for utility_idx, utility_function in enumerate(utility_functions[:-2]):
        if draw_plot:
            plt.subplot(4, 3, utility_idx + 1)
        utility_function.plot()
        fluent_utility = utility_function.apply(normalized_fluent_vector[utility_idx])
        if draw_plot:
            plt.axvline(x=normalized_fluent_vector[utility_idx])
        total_utility += fluent_utility

    if draw_plot:
        plt.show()

    return total_utility


if __name__ == '__main__':
    fluent_vector = [0.0167438, 0.533294, 0.498393, 0.48, 0.4024, 3.18043,
                     8.04958, 11.0903, 13.158, -25.2831, -17.193, -26.4704]
    utility = infer_total_utility(fluent_vector)
    print(utility)
