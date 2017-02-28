import numpy as np
from matplotlib import pyplot as plt
import fluent_data_loader as fdl
import random

class PiecewiseFunction:

    def __init__(self, n_pieces, idx=0, left_bound=-3, right_bound=3):
        self._n_pieces = n_pieces
        self._idx = idx
        self._left_bound = left_bound
        self._right_bound = right_bound
        self._joints_x = np.linspace(left_bound, right_bound, n_pieces + 1)
        self._joints_ys = []
        for pwf_idx in range(pow(2, n_pieces + 1)):
            bin_vec = [2 * int(x) for x in bin(pwf_idx)[2:]]
            padding = [0] * ((n_pieces + 1) - len(bin_vec))
            bin_vec = padding + bin_vec
            bin_vec += np.random.standard_normal(n_pieces + 1) / 4
            self._joints_ys.append(bin_vec)

    def plot(self, idx=None, plateau=5):

        if idx is None:
            idx = self._idx
        prev_joint_x = self._joints_x[0]
        prev_joint_y = self._joints_ys[idx][0]
        plt.plot([prev_joint_x-0.1, prev_joint_x], [plateau, prev_joint_y], c='r')
        for (joint_x, joint_y) in zip(self._joints_x, self._joints_ys[idx])[1:]:
            plt.plot([prev_joint_x, joint_x], [prev_joint_y, joint_y], c='r')
            prev_joint_x = joint_x
            prev_joint_y = joint_y
        plt.plot([joint_x, joint_x + 0.1], [joint_y, plateau], c='r')
        plt.xlim([-3.3, 3.3])
        plt.ylim([-2, 4])

    def apply(self, x, idx=None, plateau=5):
        if idx is None:
            idx = self._idx
        y = None
        for joint_idx, joint_x in enumerate(self._joints_x):
            if joint_x >= x:
                if joint_idx == 0:
                    # print('{} in (-inf, {}]'.format(x, joint_x))
                    if joint_x == x:
                        y = self._joints_ys[idx][joint_idx]
                    else:
                        y = plateau
                else:
                    # print('{} in ({}, {}]'.format(x, self._joints_x[joint_idx - 1], joint_x))
                    interval = joint_x - self._joints_x[joint_idx - 1]
                    contribution_curr = 1 - (joint_x - x) / interval
                    contribution_prev = 1. - contribution_curr
                    y = contribution_prev * self._joints_ys[idx][joint_idx - 1] + \
                        contribution_curr * self._joints_ys[idx][joint_idx]
                break
        if y is None:
            # print('{} in ({}, inf)'.format(x, self._joints_x[-1]))
            y = plateau
        # print 'f({}) = {}'.format(x, y)
        return y


def utility(fluent_vecs, utility_functions):
    num_examples, num_fluents = np.shape(fluent_vecs)
    total_fluent_utility = 0.
    for fluent_idx in range(num_fluents):
        avg_fluent_utility = 0.
        for fluent_val in fluent_vecs[fluent_idx, :]:
            avg_fluent_utility += utility_functions[fluent_idx].apply(fluent_val)
        avg_fluent_utility /= num_examples
        total_fluent_utility += avg_fluent_utility
    return total_fluent_utility


def fluent_utility(fluents, utility_function):
    avg_fluent_utility = 0.
    for fluent_val in fluents:
        avg_fluent_utility += utility_function.apply(fluent_val)
    avg_fluent_utility /= np.size(fluents)
    return avg_fluent_utility


def plot_utilities(utility_functions, fluents):
    plt.figure()
    for utility_idx, utility_function in enumerate(utility_functions):
        plt.subplot(4, 3, utility_idx + 1)
        plt.title('Fluent {}'.format(utility_idx))
        utility_function.plot()
        plt.scatter(fluents[:, utility_idx], [0]*np.shape(fluents)[0], marker='x')
    plt.tight_layout()
    plt.savefig('utilities.png')
    plt.show()


if __name__ == '__main__':
    n_pieces = 7

    loader = fdl.DataLoader()
    end_fluents = np.asarray(loader.get_end_fluents())
    num_fluents = np.shape(end_fluents)[1]

    best_utility_functions = []

    # for fluent_idx in range(num_fluents):
    #     best_pwf_idx = None
    #     min_fluent_utility = float('inf')
    #     for pwf_idx in range(pow(2, n_pieces + 1)):
    #         pwf = PiecewiseFunction(n_pieces=n_pieces, idx=pwf_idx)
    #         fu = fluent_utility(end_fluents[:, fluent_idx], pwf)
    #         if fu < min_fluent_utility:
    #             min_fluent_utility = fu
    #             best_pwf_idx = pwf_idx
    #     best_utility_functions.append(PiecewiseFunction(n_pieces=n_pieces, idx=best_pwf_idx))


    all_utility_functions = []
    for i in range(10000):
        if i % 1000 == 0:
            print('i', i)
        all_utility_functions.append(
            [PiecewiseFunction(n_pieces=n_pieces, idx=random.randint(0, pow(2, n_pieces + 1) - 1) if i is not 0 else 0)
             for i in range(num_fluents)])

    min_u = float('inf')
    best_utility_functions = None
    for i, utility_functions in enumerate(all_utility_functions):
        if i % 1000 == 0:
            print('i', i)
        u = utility(end_fluents, utility_functions)
        if u < min_u:
            min_u = u
            best_utility_functions = utility_functions[:]
    print('utility', min_u)

    plot_utilities(best_utility_functions, end_fluents)





