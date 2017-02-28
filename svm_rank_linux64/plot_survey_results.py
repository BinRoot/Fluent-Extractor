import numpy as np
import csv
import fluent_data_loader as fdl
from matplotlib import pyplot as plt

survey_filename = 'survey.csv'

survey_results = []

with open(survey_filename, 'rb') as csvfile:
    csvreader = csv.reader(csvfile)
    csvfile.readline()
    for row in csvreader:
        result = [-1 if field[-1] == 'A' else 1 for field in row[1:]]
        survey_results.append(result)

human_preferences = np.mean(survey_results, axis=0)
print(human_preferences)

q1 = '20:0', '20:49'
q2 = '22:0', '22:25'
q3 = '25:25', '20:49'
q4 = '28:19', '35:15'
q5 = ('5:0', '5:40'), ('6:0', '6:28')
q6 = ('16:22', '16:29'), ('25:25', '25:32')
q7 = ('13:12', '13:20'), ('17:13', '17:22')

loader = fdl.DataLoader()

max_val = np.max(loader._values)
min_val = np.min(loader._values)
print(min_val, max_val)

plt.figure()
all_robot_preferences = []
for i in range(1, 45 + 1):
    value_model_file = 'train_datasets/value_model_{}'.format(i)

    robot_preferences = []
    value_q1 = loader.get_value_from_meta(q1[0], model_file=value_model_file), \
               loader.get_value_from_meta(q1[1], model_file=value_model_file)
    robot_preferences.append(value_q1[1] - value_q1[0])

    value_q2 = loader.get_value_from_meta(q2[0], model_file=value_model_file), \
               loader.get_value_from_meta(q2[1], model_file=value_model_file)
    robot_preferences.append(value_q2[1] - value_q2[0])

    value_q3 = loader.get_value_from_meta(q3[0], model_file=value_model_file), \
               loader.get_value_from_meta(q3[1], model_file=value_model_file)
    robot_preferences.append(value_q3[1] - value_q3[0])

    value_q4 = loader.get_value_from_meta(q4[0], model_file=value_model_file), \
               loader.get_value_from_meta(q4[1], model_file=value_model_file)
    robot_preferences.append(value_q4[1] - value_q4[0])

    value_q5a = loader.get_value_from_meta(q5[0][0], model_file=value_model_file), \
                loader.get_value_from_meta(q5[0][1], model_file=value_model_file)
    value_q5b = loader.get_value_from_meta(q5[1][0], model_file=value_model_file), \
                loader.get_value_from_meta(q5[1][1], model_file=value_model_file)
    robot_preferences.append((value_q5b[1] - value_q5b[0]) - (value_q5a[1] - value_q5a[0]))

    value_q6a = loader.get_value_from_meta(q6[0][0], model_file=value_model_file), \
                loader.get_value_from_meta(q6[0][1], model_file=value_model_file)
    value_q6b = loader.get_value_from_meta(q6[1][0], model_file=value_model_file), \
                loader.get_value_from_meta(q6[1][1], model_file=value_model_file)
    robot_preferences.append((value_q6b[1] - value_q6b[0]) - (value_q6a[1] - value_q6a[0]))

    value_q7a = loader.get_value_from_meta(q7[0][0], model_file=value_model_file), \
                loader.get_value_from_meta(q7[0][1], model_file=value_model_file)
    value_q7b = loader.get_value_from_meta(q7[1][0], model_file=value_model_file), \
                loader.get_value_from_meta(q7[1][1], model_file=value_model_file)
    robot_preferences.append((value_q7b[1] - value_q7b[0]) - (value_q7a[1] - value_q7a[0]))

    robot_preferences = np.array(robot_preferences)
    robot_preferences[0:4] /= np.max(robot_preferences[0:4])
    robot_preferences[4:] /= (max(np.max(robot_preferences[4:]), -np.min(robot_preferences[4:])) / 0.618)
    all_robot_preferences.append(robot_preferences)
    print(i, robot_preferences)

    plt.subplot(9, 5, i)
    bar_index = np.arange(len(human_preferences))
    bar_width = 0.35
    rects_1 = plt.barh(bar_index - bar_width/2,
                      np.fliplr([human_preferences])[0],
                      bar_width,
                      color='b')
    rects_2 = plt.barh(bar_index + bar_width/2,
                      np.fliplr([robot_preferences])[0],
                      bar_width,
                      color='r')
    plt.yticks(bar_index + bar_width / 2, ('G', 'F', 'E', 'D', 'C', 'B', 'A'))
    plt.xticks([])
# plt.tight_layout()

all_robot_preferences = np.asarray(all_robot_preferences)

plt.figure()
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for i in range(7):
    plt.plot(np.arange(1, 46), all_robot_preferences[:, i],
             lw=2.5,
             label='Decision {}'.format(labels[i]))
plt.xlabel('Number of videos')
plt.ylabel('Preference')
plt.ylim([-1.1, 1.1])
plt.legend(loc=2)
plt.show()



