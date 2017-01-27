import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize

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


def load_action_dataset(filename):
    actions = defaultdict(lambda: [])

    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        line_parts = line.split(',')
        if len(line_parts) == 1:
            continue
        action_label = line_parts[0]
        metas = line_parts[1].split('-')
        start_meta, end_meta = metas[0], metas[1][:-1]
        print(start_meta, end_meta)
        actions[action_label].append(line_parts[1][:-1])
    return actions



def plot_actions_per_video(all_indices, dataset_normalized):
    plt.figure()
    plt.title('Fluent changes in cloth folding videos')
    for idx, indices in enumerate(all_indices):
        plt.subplot(9, 5, idx + 1)
        diffs = process_video(dataset_normalized[indices, :])
        plt.imshow(diffs, interpolation='nearest')
        plt.yticks(list(range(len(indices) - 1)), [meta_info[indices_val] for indices_val in indices[1:]])
    plt.show()


def plot_actions(dataset_normalized, meta_info, actions):
    action_to_df = {}
    meta_to_label = {}
    for action_label, metas in actions.items():
        for meta in metas:
            meta_to_label[meta] = action_label

    action_data = defaultdict(lambda: [])
    action_data_for_plot = defaultdict(lambda: [])
    for idx, indices in enumerate(all_indices):
        diffs = process_video(dataset_normalized[indices, :])
        # each row of diffs, gives a diff, meta, and precondition
        # make a map action_label -> [[precondition, diff], ...]
        prev_meta = meta_info[indices[0]]
        prev_fluent = dataset_normalized[indices[0]]
        for j, i in enumerate(indices[1:]):
            meta = meta_info[i]
            fluent = dataset_normalized[i, :]
            meta_diff = '{}-{}'.format(prev_meta, meta)
            if meta_diff in meta_to_label:
                action_label = meta_to_label[meta_diff]
                action_data_for_plot[action_label].append(np.hstack((prev_fluent, [float("inf")], diffs[j], [float("inf")], fluent)))
                action_data[action_label].append(diffs[j])
            prev_meta = meta
            prev_fluent = fluent[:]

    plt.figure()
    plt.title('Fluent changes caused by actions')
    for idx, (action_label, metas) in enumerate(actions.items()):
        plt.subplot(4, 3, idx + 1)
        plt.title(action_label)
        plt.imshow(action_data_for_plot[action_label], vmin=-3, vmax=3, interpolation='nearest')
        vars = np.var(action_data[action_label], axis=0)
        means = np.abs(np.mean(action_data[action_label], axis=0))
        print(action_label)
        print('vars', vars)
        print('means', means)

        x0 = np.random.standard_normal(len(vars))

        def relevant_variance(w):
            return np.linalg.norm(w * vars, ord=1) #+ 100 * np.linalg.norm(w, ord=1)

        def relevant_variance_backup(w):
            return np.linalg.norm(w * vars/means) + np.linalg.norm(w, ord=1)

        cons = ({
            'type': 'ineq',
            'fun': lambda x: -x + 1
        }, {
            'type': 'ineq',
            'fun': lambda x: x
        }, {
            'type': 'ineq',
            'fun': lambda x: np.linalg.norm(x, ord=1) - 6
        })

        res = minimize(relevant_variance, x0, method='SLSQP', constraints=cons)
        solved_x = res['x']
        solved_x[res['x'] < 1e-3] = 0
        print('solved x', solved_x, res['success'], res['message'])
        action_to_df[action_label] = solved_x * means

    plt.tight_layout()
    # plt.show()
    return action_to_df


def compute_heuristic_cost(states, end_fluent):
    min_cost_idx = None
    min_heuristic_cost = float('inf')
    for idx, (hist, state) in enumerate(states):
        heuristic_cost = np.linalg.norm(state - end_fluent)
        if heuristic_cost < min_heuristic_cost:
            min_heuristic_cost = heuristic_cost
            min_cost_idx = idx
    return min_cost_idx, min_heuristic_cost


def infer_actions(start_fluent, end_fluent, action_to_df):
    states = [([], start_fluent)]
    count = 0
    while count < 1000:
        min_cost_idx, cost = compute_heuristic_cost(states, end_fluent)
        fluent_hist, fluent_val = states[min_cost_idx]
        del states[min_cost_idx]
        for action_name, action_f in action_to_df.items():
            new_state = (fluent_hist + [action_name], fluent_val + action_f)
            states.append(new_state)
        count += 1
    min_cost_idx, cost = compute_heuristic_cost(states, end_fluent)
    print(cost, states[min_cost_idx])


if __name__ == '__main__':
    actions = load_action_dataset('action_dataset.txt')

    dataset, meta_info, all_indices = load_dataset('value_train.dat')
    scaler = preprocessing.StandardScaler().fit(dataset)
    dataset_normalized = scaler.transform(dataset)

    # plot_actions_per_video(all_indices, dataset_normalized)
    action_to_df = plot_actions(dataset_normalized, meta_info, actions)

    idx = 0
    start_fluent = dataset_normalized[all_indices[idx][0], :]
    end_fluent = dataset_normalized[all_indices[idx][-1], :]
    infer_actions(start_fluent, end_fluent, action_to_df)



