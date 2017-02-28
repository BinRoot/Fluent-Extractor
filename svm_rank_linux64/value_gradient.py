import numpy as np
import subprocess, os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib.pyplot import cm

def load_dataset(filename):
    """
    :param filename: *.dat file
    :return: numpy matrix samples x features, and string list of meta info
    """
    with open(filename) as f:
        lines = f.readlines()

    dataset = []
    meta_info = []
    all_indices = []
    # [[0, 1, 2], [3, 4], ...]
    # get corresponding mds points

    indices = []
    prev_vid_idx = None
    for line_idx, line in enumerate(lines):
        line_arr = line.strip().split(' ')
        img_idx = line_arr[0]
        vid_idx = line_arr[1].split(':')[1]
        if prev_vid_idx is None:
            indices = []
        elif prev_vid_idx != vid_idx or line_idx == len(lines) - 1:
            all_indices.append(indices)
            indices = []
        indices.append(line_idx)

        meta_info.append('{}:{}'.format(vid_idx, img_idx))
        feature_vector = []
        for dim_str in line_arr[2:]:
            feature_vector.append(float(dim_str.split(':')[1]))
        dataset.append(feature_vector)
        prev_vid_idx = vid_idx
    dataset = np.asarray(dataset)

    return dataset, meta_info, all_indices


def save_to_dat(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            v_arr = []
            for i, v in enumerate(row):
                v_arr.append('{}:{}'.format(i + 1, v))
            f_str = ' '.join(v_arr)
            row_str = '0 qid:1 {}\n'.format(f_str)
            f.write(row_str)


def compute_values(train_filename):
    prediction_file = 'value_predictions'
    infer_cmd = "./svm_rank_classify {} value2_model {}".format(train_filename, prediction_file)
    process = subprocess.Popen(infer_cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()

    with open(os.path.join(prediction_file), 'r') as f:
        val_lines = f.readlines()

    vals = map(float, val_lines)
    return np.asarray(vals)


def step_ascent(start_f, scaler):
    num_samples = 10000
    ds = np.random.normal(0, 0.01, [num_samples, np.size(start_f)])
    sampled_fs = start_f + ds
    sampled_fs_unnormalized = scaler.inverse_transform(sampled_fs)
    save_to_dat('value_train_samples.dat', sampled_fs_unnormalized)
    predictions = compute_values('value_train_samples.dat')
    max_idx = np.argmax(predictions)
    max_val = predictions[max_idx]
    return sampled_fs[max_idx], max_val


def start_ascent(indices, dataset, meta_info):
    scaler = preprocessing.StandardScaler().fit(dataset)
    dataset_normalized = scaler.transform(dataset)
    start_idx = indices[0]
    start_f = dataset_normalized[start_idx, :]
    print('starting with {}'.format(start_f))

    measurement_trace = []
    plt.figure()
    for i in range(10000):
        next_f, next_val = step_ascent(start_f, scaler)
        print('Value {}'.format(next_val))
        # how close is it to dataset
        dist = (dataset_normalized - next_f) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        dataset_idx = np.argmin(dist)
        print('dist to closest fluent is {} ({})'.format(dist[dataset_idx], meta_info[dataset_idx]))
        start_f = next_f
        measurement_trace.append([next_val, dist[dataset_idx], meta_info[dataset_idx]])
        measurement_trace_np = np.asarray(measurement_trace)
        plt.clf()
        plt.ylabel('Value')
        plt.plot(measurement_trace_np[:, 0], marker='o', color='b', linestyle='--')
        for row_idx, (row_val, row_dist, row_meta) in enumerate(measurement_trace):
            if row_dist < 0.5:
                plt.annotate(row_meta,
                             xy=(row_idx, row_val),
                             xytext=(row_idx-0.5, row_val+0.5))
        ax2 = plt.twinx()
        ax2.plot(measurement_trace_np[:, 1], color='red')
        plt.pause(0.05)


def start_interpolation(start_f, end_f):
    num_steps = 100
    d = (end_f - start_f) / float(num_steps)
    test_fs = [start_f + d*i for i in range(num_steps)]
    save_to_dat('value_train_samples.dat', test_fs)
    predictions = compute_values('value_train_samples.dat')
    return predictions


def start_interpolations(indices, dataset, meta_info):
    start_idx = indices[0]
    plot_x_size = 0
    list_of_plots = []
    for end_idx in indices[1:]:
        predictions = start_interpolation(dataset[start_idx, :], dataset[end_idx, :])
        xs = list(range(plot_x_size, plot_x_size + len(predictions)))
        list_of_plots.append((xs, predictions))
        plot_x_size += len(predictions)
        start_idx = end_idx
    return list_of_plots


def plot_all_interpolations(all_indices, dataset, meta_info):
    plt.figure()
    plt.suptitle('Value over time in cloth folding videos')
    colors = cm.rainbow(np.linspace(0, 1, 7))
    for i in range(len(all_indices)):
        ax2 = plt.subplot(9, 5, i + 1)
        subplot_title = 'Video {}'.format(i + 1)
        print('plotting {}'.format(subplot_title))
        # plt.title(subplot_title)
        plt.text(0.5, 0.8, subplot_title,
                 horizontalalignment='center',
                 fontsize=8,
                 transform=ax2.transAxes)
        plt.xticks([])
        # plt.yticks(np.arange(-10, 25, 5))
        # plt.ylim([-10, 25])
        plt.yticks(np.arange(-0.5, 1.5, 0.5))
        plt.ylim([-0.5, 1.5])
        list_of_plots = \
            start_interpolations(all_indices[i], dataset, meta_info)
        for plot_idx, (xs, predictions) in enumerate(list_of_plots):
            plt.plot(xs, predictions, c=colors[plot_idx], lw=3)
        plt.pause(0.2)
    plt.show()

if __name__ == '__main__':
    dataset, meta_info, all_indices = load_dataset('value_train.dat')
    # start_ascent(all_indices[0], dataset, meta_info)
    plot_all_interpolations(all_indices, dataset, meta_info)

