'''
Author: Yunzhong He

* This script splits all the data into train, validate and test sets. And performs
a grid search for rank-svm parameters that does best on the validation set.
* It saves the trained model for every parameter sets along with its outputs.

'''

import random
import subprocess
import os

NUM_VID = 45
NUM_TRAIN = 27
NUM_CV = 9
NUM_TEST = 9
random.seed(1234) # Change the seed for a different split

def write_rows(filename, data):
    file = open(filename, 'w+')
    for rows in data:
        for row in rows:
            print >> file, row,
    file.close()

def run_cmd(command):
    subprocess.call(command, shell=True)

def grid_search(train, validate):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for c in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
        print 'c = %f' % (c)
        model_name = 'model_c_%f' % (c)
        run_cmd('../svm_rank_learn -c %f -t 1 -d 3 -v 3 train.dat ./models/%s' 
            % (c, model_name))
        output = run_cmd('../svm_rank_classify validate.dat ./models/%s > ./models/%s.report' 
            % (model_name, model_name))

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
    test_indices = indices[NUM_TRAIN+NUM_CV:]

    train_indices.sort()
    validate_indices.sort()
    test_indices.sort()

    train = [data[qid] for qid in train_indices]
    validate = [data[qid] for qid in validate_indices]
    test = [data[qid] for qid in test_indices]

    assert len(train) == NUM_TRAIN and len(validate) == NUM_CV and len(test) == NUM_TEST

    write_rows('train.dat', train)
    write_rows('validate.dat', validate)
    write_rows('test.dat', test)

    # Prepare model directory
    if not os.path.exists('./models'):
        os.makedirs('./models')


    # Grid search of optimal parameters
    grid_search(train, validate)
