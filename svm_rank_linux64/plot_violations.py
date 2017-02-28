import fluent_data_loader as fdl
import numpy as np
import os
import subprocess
from scipy.stats import rankdata, pearsonr
from matplotlib import pyplot as plt

def train_models(directory):
    for i in range(1, 45 + 1):
        filename_in = 'value2_train_{}.dat'.format(i)
        relative_path_in = os.path.join(directory, filename_in)

        filename_out = 'value2_model_{}'.format(i)
        relative_path_out = os.path.join(directory, filename_out)

        print(relative_path_in, relative_path_out)
        learn_cmd = "./svm_rank_learn -c 20 -t 1 -d 3 {} {}".format(relative_path_in, relative_path_out)
        process = subprocess.Popen(learn_cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()

def count_violations(all_indices, values):
    avg_r = 0.
    for indices in all_indices:
        sub_values = values[indices]
        indices_rank = rankdata(indices, method='ordinal')
        values_rank = rankdata(sub_values, method='ordinal')
        r = pearsonr(indices_rank, values_rank)[0]
        avg_r += r
    return avg_r / len(all_indices)

def test_models(directory):
    violations = []
    loader = fdl.DataLoader()
    for i in range(1, 45 + 1):
        model_filename = os.path.join(directory, 'value{}_model_{}'.format('2' if directory[-1] == '2' else '', i))
        values = loader.load_values(model_filename=model_filename)
        violation = count_violations(loader._all_indices[30:], values)
        print(i, violation)
        violations.append(violation)
    return violations

def test_sparse():
    directory = 'train_datasets2'
    vs = test_models(directory)
    np.save('sparse_violations.npy', vs)
    return vs

def test_dense():
    directory = 'train_datasets'
    vs = test_models(directory)
    np.save('dense_violations.npy', vs)
    return vs

if __name__ == '__main__':
    directory = 'train_datasets2'
    vs = test_sparse()
    vd = test_dense()
    plt.figure()
    plt.plot(vs[0:30], marker='x', lw=3)
    plt.plot(vd[0:30], marker='+', lw=3)
    plt.show()