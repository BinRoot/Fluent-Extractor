import plot_minimax_survey
import fluent_data_loader as fdl
from matplotlib import pyplot as plt

loader = fdl.DataLoader()
utility_functions = plot_minimax_survey.get_utility_functions()


all_indices = loader._all_indices
indices = all_indices[0]
fluent_vec1 = loader._dataset_normalized[indices[0]]
fluent_vec2 = loader._dataset_normalized[indices[1]]
fluent_vec3 = loader._dataset_normalized[indices[2]]
fluent_vec4 = loader._dataset_normalized[indices[3]]

plt.figure()

for utility_idx, utility_function in enumerate(utility_functions):
    plt.subplot(4, 3, utility_idx + 1)
    utility_function.plot()

    plt.scatter(fluent_vec1[utility_idx], 0, c='k', marker='$ 1 $', s=100)
    fluent_cost1 = utility_function.apply(fluent_vec1[utility_idx])
    plt.text(0, 3, '{:.2f}'.format(fluent_cost1))

    plt.scatter(fluent_vec2[utility_idx], 0, c='k', marker='$ 2 $', s=100)
    fluent_cost2 = utility_function.apply(fluent_vec2[utility_idx])
    plt.text(0, 2, '{:.2f}'.format(fluent_cost2))

    plt.scatter(fluent_vec3[utility_idx], 0, c='k', marker='$ 3 $', s=100)
    fluent_cost3 = utility_function.apply(fluent_vec3[utility_idx])
    plt.text(0, 1, '{:.2f}'.format(fluent_cost3))

plt.show()


