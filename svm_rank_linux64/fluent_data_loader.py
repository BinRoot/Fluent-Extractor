import sys, os
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
import subprocess


class DataLoader:
    def __init__(self,
                 action_dataset_filename='action_dataset.txt',
                 value_dataset_filename='value_train.dat'):
        self._actions = self._load_action_dataset(action_dataset_filename)
        self._dataset, self._meta_info, self._all_indices = \
            self._load_dataset(value_dataset_filename)
        self._scaler = preprocessing.StandardScaler().fit(self._dataset)
        self._dataset_normalized = self._scaler.transform(self._dataset)
        self._meta_to_label = {}
        for action_label, metas in self._actions.items():
            for meta in metas:
                self._meta_to_label[meta] = action_label
        self._values = self.load_values()

    def normalize(self, fluent_vector):
        fluent_vector = np.array(fluent_vector)
        return self._scaler.transform(fluent_vector.reshape(1, -1))[0]

    def get_fluents_from_meta(self, meta):
        idx = self._meta_info.index(meta)
        if idx == -1:
            return None
        return self._dataset_normalized[idx, :]

    def get_value_from_meta(self, meta, model_file=None):
        idx = self._meta_info.index(meta)
        if idx == -1:
            return None

        values = self._values \
            if model_file is None \
            else self.load_values(model_filename=model_file)

        return values[idx]

    def load_action_data(self):
        action_data_f_start = defaultdict(lambda: [])
        action_data_f_end = defaultdict(lambda: [])
        for idx, indices in enumerate(self._all_indices):
            prev_meta = self._meta_info[indices[0]]
            prev_fluent = self._dataset_normalized[indices[0]]
            for j, i in enumerate(indices[1:]):
                meta = self._meta_info[i]
                fluent = self._dataset_normalized[i, :]
                meta_diff = '{}-{}'.format(prev_meta, meta)
                if meta_diff in self._meta_to_label:
                    action_label = self._meta_to_label[meta_diff]
                    action_data_f_start[action_label].append(prev_fluent)
                    action_data_f_end[action_label].append(fluent)
                prev_meta = meta
                prev_fluent = fluent[:]
        action_data = {}
        for action_label in action_data_f_start.keys():
            action_data_f_start[action_label] = np.asarray(action_data_f_start[action_label])
            action_data_f_end[action_label] = np.asarray(action_data_f_end[action_label])
            action_data[action_label] = (action_data_f_start[action_label], action_data_f_end[action_label] - action_data_f_start[action_label])
        return action_data

    def get_values(self):
        return self._values

    @staticmethod
    def load_values(train_filename='value_train.dat', model_filename='value_model'):
        prediction_file = 'value_predictions'
        infer_cmd = "./svm_rank_classify {} {} {}".format(train_filename, model_filename, prediction_file)
        process = subprocess.Popen(infer_cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

        with open(os.path.join(prediction_file), 'r') as f:
            val_lines = f.readlines()

        vals = map(float, val_lines)
        return np.asarray(vals)

    def get_start_fluents(self):
        start_fluents = []
        for indices in self._all_indices:
            start_idx = indices[0]
            start_fluent = self._dataset_normalized[start_idx, :]
            start_fluents.append(start_fluent)
        return start_fluents

    def get_end_fluents(self):
        end_fluents = []
        for indices in self._all_indices:
            end_idx = indices[-1]
            end_fluent = self._dataset_normalized[end_idx, :]
            end_fluents.append(end_fluent)
        return end_fluents

    def get_goal(self, vid_id):
        """
        :param vid_id: video id
        :return: tuple (start_fluent, end_fluent)
        """
        indices = self._all_indices[vid_id]
        start_idx, end_idx = indices[0], indices[-1]
        start_fluent = self._dataset_normalized[start_idx, :]
        end_fluent = self._dataset_normalized[end_idx, :]
        return start_fluent, end_fluent

    @staticmethod
    def _load_dataset(filename):
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

    @staticmethod
    def _load_action_dataset(filename):
        actions = defaultdict(lambda: [])

        with open(filename) as f:
            lines = f.readlines()

        for line in lines:
            line_parts = line.split(',')
            if len(line_parts) == 1:
                continue
            action_label = line_parts[0]
            actions[action_label].append(line_parts[1][:-1])
        return actions

if __name__ == '__main__':
    """
    Demonstrates how to use DataLoader
    """

    loader = DataLoader()
    start_fluent, end_fluent = loader.get_goal(0)
    print('start', start_fluent)
    print('end', end_fluent)
    action_data = loader.load_action_data()

    plt.figure()
    for idx, (action_label, (start_examples, end_examples)) in enumerate(action_data.items()):
        plt.subplot(4, 3, idx + 1)
        plt.title(action_label)
        column_vec = float('inf') * np.ones((np.shape(start_examples)[0], 1))
        viz_matrix = np.hstack((start_examples, column_vec, end_examples))
        plt.imshow(viz_matrix, vmin=-3, vmax=3, interpolation='nearest')
    plt.tight_layout()
    plt.show()
