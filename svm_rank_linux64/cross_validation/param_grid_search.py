'''
Author: Yunzhong He

* This script splits all the data into train, validate and test sets. And performs
a grid search for rank-svm parameters that does best on the validation set.
* It saves the trained model for every parameter sets along with its outputs.

'''

import random
import subprocess
import os
import numpy as np

NUM_VID = 45
NUM_TRAIN = int(0.7 * NUM_VID)
NUM_CV = NUM_VID - NUM_TRAIN
random.seed(1234) # Change the seed for a different split

def write_rows(filename, data):
    file = open(filename, 'w+')
    for rows in data:
        for row in rows:
            print >> file, row,
    file.close()


def run_cmd(command):
    subprocess.call(command, shell=True)


def grid_search():
    for c in np.logspace(-3, 3, 7):
        for t in [1, 2, 3]:
            if t == 1:  # polynomial (s a*b+r)^d
                for d in [2, 3, 4]:
                    for r in np.linspace(-10, 10, 3):
                        model_name = 'model_c_{}_t_{}_d_{}_r_{}'.format(c, t, d, r)
                        run_cmd('../svm_rank_learn -c {} -t {} -d {} -r {} -v 1 train.dat ./models/{}'
                                .format(c, t, d, r, model_name))
                        run_cmd('../svm_rank_classify validate.dat ./models/{} > ./models/{}.report'
                                .format(model_name, model_name))
            elif t == 2:  # rbf exp(-gamma ||a-b||^2)
                for g in np.logspace(-3, 3, 7):
                    model_name = 'model_c_{}_t_{}_g_{}'.format(c, t, g)
                    run_cmd('../svm_rank_learn -c {} -t {} -g {} -v 1 train.dat ./models/{}'
                            .format(c, t, d, model_name))
                    run_cmd('../svm_rank_classify validate.dat ./models/{} > ./models/{}.report'
                            .format(model_name, model_name))
            elif t == 3:  # sigmoid tanh(s a*b + r)
                for r in np.linspace(-10, 10, 3):
                    for s in np.linspace(-1, 1, 4):
                        model_name = 'model_c_{}_t_{}_r_{}_s_{}'.format(c, t, r, s)
                        run_cmd('../svm_rank_learn -c {} -t {} -r {} -s {} -v 1 train.dat ./models/{}'
                                .format(c, t, r, s, model_name))
                        run_cmd('../svm_rank_classify validate.dat ./models/{} > ./models/{}.report'
                                .format(model_name, model_name))

if __name__ == '__main__':
    # Group fluents by video id
    data = dict()
    file = open('../value_train.dat')
    for line in file:
        qid = int(line.split(' ')[1][4:])
        if not qid in data:
            data[qid] = [line]
        else:
            data[qid].append(line)
    file.close()

    # Shuffle and split
    indices = range(1, NUM_VID+1)
    random.shuffle(indices)

    train_indices = indices[0:NUM_TRAIN]
    validate_indices = indices[NUM_TRAIN:NUM_TRAIN+NUM_CV]

    train_indices.sort()
    validate_indices.sort()

    train = [data[qid] for qid in train_indices]
    validate = [data[qid] for qid in validate_indices]

    assert len(train) == NUM_TRAIN and len(validate) == NUM_CV

    write_rows('train.dat', train)
    write_rows('validate.dat', validate)

    # Prepare model directory
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Grid search of optimal parameters
    grid_search()
