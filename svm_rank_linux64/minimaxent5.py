import numpy as np
from matplotlib import pyplot as plt
import fluent_data_loader as fdl
import random


class PiecewiseFunction:

    def __init__(self, n_pieces, idx=None, left_bound=-3, right_bound=3):
        self._idx = 0 if idx is None else idx
        self._n_pieces = n_pieces
        self._left_bound = left_bound
        self._right_bound = right_bound
        self._joints_x = np.linspace(left_bound, right_bound, n_pieces + 1)
        self._joints_ys = []
        if idx is None:
            for pwf_idx in range(pow(2, n_pieces + 1)):
                bin_vec = [2 * int(x) for x in bin(pwf_idx)[2:]]
                padding = [0] * ((n_pieces + 1) - len(bin_vec))
                bin_vec = padding + bin_vec
                bin_vec += np.random.standard_normal(n_pieces + 1) / 4
                self._joints_ys.append(bin_vec)
        else:
            bin_vec = [2 * int(x) for x in bin(idx)[2:]]
            padding = [0] * ((n_pieces + 1) - len(bin_vec))
            bin_vec = padding + bin_vec
            bin_vec += np.random.standard_normal(n_pieces + 1) / 4
            self._joints_y = bin_vec

    def __str__(self):
        joints_y = None
        if self._joints_y is not None:
            joints_y = self._joints_y
        else:
            joints_y = self._joints_ys[self._idx]

        return '[{}]'.format(','.join(map(str, joints_y)))

    def plot(self, idx=None, plateau=5):
        joints_y = None
        if idx is not None:
            joints_y = self._joints_ys[idx]
        elif self._joints_y is not None:
            joints_y = self._joints_y
        else:
            joints_y = self._joints_ys[self._idx]

        prev_joint_x = self._joints_x[0]
        prev_joint_y = joints_y[0]
        plt.plot([prev_joint_x-0.1, prev_joint_x], [plateau, prev_joint_y], c='r')
        for (joint_x, joint_y) in zip(self._joints_x, joints_y)[1:]:
            plt.plot([prev_joint_x, joint_x], [prev_joint_y, joint_y], c='r')
            prev_joint_x = joint_x
            prev_joint_y = joint_y
        plt.plot([joint_x, joint_x + 0.1], [joint_y, plateau], c='r')
        plt.xlim([-3.3, 3.3])
        plt.ylim([-2, 4])

    def apply(self, x, idx=None, plateau=5):
        joints_y = None
        if idx is not None:
            joints_y = self._joints_ys[idx]
        elif self._joints_y is not None:
            joints_y = self._joints_y
        else:
            joints_y = self._joints_ys[self._idx]

        y = None
        for joint_idx, joint_x in enumerate(self._joints_x):
            if joint_x >= x:
                if joint_idx == 0:
                    # print('{} in (-inf, {}]'.format(x, joint_x))
                    if joint_x == x:
                        y = joints_y[joint_idx]
                    else:
                        y = plateau
                else:
                    # print('{} in ({}, {}]'.format(x, self._joints_x[joint_idx - 1], joint_x))
                    interval = joint_x - self._joints_x[joint_idx - 1]
                    contribution_curr = 1 - (joint_x - x) / interval
                    contribution_prev = 1. - contribution_curr
                    y = contribution_prev * joints_y[joint_idx - 1] + \
                        contribution_curr * joints_y[joint_idx]
                break
        if y is None:
            # print('{} in ({}, inf)'.format(x, self._joints_x[-1]))
            y = plateau
        # print 'f({}) = {}'.format(x, y)
        return y


def utility(fluent_vec, utility_functions):
    total_fluent_utility = 0.
    for fluent_idx, fluent_val in enumerate(fluent_vec):
        total_fluent_utility += utility_functions[fluent_idx].apply(fluent_val)
    return total_fluent_utility


def prob(fluent_vec, utility_functions):
    return np.exp(-utility(fluent_vec, utility_functions))


def ranking_slack(start_fluents, end_fluents, utility_functions):
    agreements = []
    num_videos = np.shape(start_fluents)[0]
    for video_idx in range(num_videos):
        agreement = 0
        for fluent_idx, pwf in enumerate(utility_functions):
            agreement += pwf.apply(start_fluents[video_idx, fluent_idx]) - pwf.apply(end_fluents[video_idx, fluent_idx])
            agreements.append(agreement)
    return num_videos - np.sum(agreements)

if __name__ == '__main__':
    n_pieces = 5

    loader = fdl.DataLoader()
    end_fluents = np.asarray(loader.get_end_fluents())
    start_fluents = np.asarray(loader.get_start_fluents())
    num_fluents = np.shape(end_fluents)[1]

    best_utility_functions = [PiecewiseFunction(n_pieces=n_pieces, idx=random.randint(0, pow(2, n_pieces + 1) - 1)) for i in range(num_fluents)]
    min_slack = float('inf')
    for outer_loop_idx in range(10):
        print('epoch {}'.format(outer_loop_idx + 1))
        for utility_idx in range(num_fluents):
            print('Fluctuating fluent {}'.format(utility_idx + 1))
            n_steps = 100
            for i in range(n_steps):
                if i % 1000 == 0:
                    print('progress {}/{}'.format(i, n_steps))
                utility_functions = [PiecewiseFunction(n_pieces=n_pieces, idx=random.randint(0, pow(2, n_pieces + 1) - 1))
                                     if i == utility_idx
                                     else best_utility_functions[i]
                                     for i in range(num_fluents)]
                slack = ranking_slack(start_fluents, end_fluents, utility_functions)
                if slack < min_slack:
                    min_slack = slack
                    print('new min slack: {}'.format(min_slack))
                    best_utility_functions = utility_functions[:]

    print(min_slack)
    plt.figure()
    for i, utility_function in enumerate(best_utility_functions):
        plt.subplot(4, 3, i + 1)
        plt.title(i)
        utility_function.plot()
        plt.scatter(start_fluents[:, i], [0]*np.shape(start_fluents[:, i])[0], marker='x', color='r')
        plt.scatter(end_fluents[:, i], [0] * np.shape(end_fluents[:, i])[0], marker='+', color='g')
        print('params{} = {}'.format(i+1, utility_function))
    plt.savefig('chart.png')
    plt.show()
