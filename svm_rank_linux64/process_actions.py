import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


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
        elif prev_vid_idx != vid_idx:
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


def process_video(dataset):
    """
    Process sequence of fluent changes in a single demonstration
    :param dataset: np array of size (n_obs, n_fluents)
    """

    if dataset.shape[0] < 2:
        print('Could not process video. Only {} fluents. Need at least 2.'.format(dataset.shape[0]))

    diffs = []
    prev_fluent_vec = dataset[0, :]
    for fluent_vec in dataset[1:, :]:
        diff = fluent_vec - prev_fluent_vec
        diffs.append(diff)
        prev_fluent_vec = fluent_vec

    return diffs

if __name__ == '__main__':
    dataset, meta_info, all_indices = load_dataset('value_train.dat')

    scaler = preprocessing.StandardScaler().fit(dataset)
    dataset_normalized = scaler.transform(dataset)

    plt.figure()
    plt.title('Fluent changes in cloth folding videos')
    for idx, indices in enumerate(all_indices):
        plt.subplot(9, 5, idx + 1)
        diffs = process_video(dataset_normalized[indices, :])
        plt.imshow(diffs, interpolation='nearest')
        plt.yticks(list(range(len(indices) - 1)), [meta_info[indices_val] for indices_val in indices[1:]])
    plt.show()
