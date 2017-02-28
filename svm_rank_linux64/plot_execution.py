from matplotlib import pyplot as plt
import numpy as np

plt.figure()

group_labels = ('t-shirt', 'towel', 'long-sleeve', 'shorts', 'pants', 'scarf')

gt_ex1 = [20, 20, 20, 25, 20, 20]
gt_ex2 = [21, 23, 22, 25, 25, 21]
gt_ex3 = [20, 23, 20, 24, 20, 20]
gt_experiments = np.asarray([gt_ex1, gt_ex2, gt_ex3])
gt_vals = np.mean(gt_experiments, axis=0)
gt_stds = np.std(gt_experiments, axis=0)

act_ex1 = [10, 15, 10, 4, 10, 3]
act_ex2 = [15, 20, 15, 2, 16, 3]
act_ex3 = [20, 20, 10, 5, 16, 1]
act_experiments = np.asarray([act_ex1, act_ex2, act_ex3])
act_vals = np.mean(act_experiments, axis=0)
act_stds = np.std(act_experiments, axis=0)


p_vals = [18, 19, 19, 14, 21, 19.1]

index = np.arange(len(group_labels))
bar_width = 0.2
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index - bar_width, gt_vals, bar_width,
                 color='g',
                 yerr=gt_stds,
                 error_kw=error_config,
                 alpha=0.9,
                 label='Ground truth')

rects2 = plt.bar(index, p_vals, bar_width,
                 alpha=0.3,
                 color='r',
                 label='Predicted utility gain',
                 hatch='/')

rects3 = plt.bar(index + bar_width, act_vals, bar_width,
                 alpha=0.5,
                 color='b',
                 yerr=act_stds,
                 error_kw=error_config,
                 label='Actual utility gain',
                 hatch='\\')


plt.xlabel('Article of clothing')
plt.ylabel('Increase in utility')
plt.title('Robot Execution Performance')
plt.xticks(index + bar_width / 3, group_labels)
plt.legend(loc=9)

plt.tight_layout()
plt.ylim([0, 35])
plt.show()