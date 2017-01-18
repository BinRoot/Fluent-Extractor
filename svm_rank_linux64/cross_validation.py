import sklearn.manifold
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import subprocess, os
import sklearn.cluster
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from value_viz import load_dataset
from value_viz import compute_values

# X: 270, Y: 270, Z: 270
# data_x (270x2), data_y (270)
# train_x (189x2), train_y (189)  (70%)
# test_x (81x2), test_y (81)
def cross_validate(X, Y, Z, testing):
	data_x = [[x, y] for x, y in zip(X, Y)]
	data_y = Z
	train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=testing, random_state=0)
	return (train_x, test_x, train_y, test_y)

def mse(pred, test):
	return mean_squared_error(pred, test)

mds = sklearn.manifold.MDS(n_components=2)
dataset, meta_info, all_indices = load_dataset('value_train.dat')
coordinates = mds.fit_transform(dataset)
X = coordinates[:, 0]
Y = coordinates[:, 1]
Z = compute_values('value_train.dat')

testing = 0.3
train_x, test_x, train_y, test_y = cross_validate(X, Y, Z, testing)

# print len(train_x)
# print len(test_x)
# print len(train_y)
# print len(test_y)

# Cross validation variables: C, gamma
C = [1e-2, 1e-1, 1e0, 1e1, 1e2]
gamma = [1e-2, 1e-1, 1e0, 1e1, 1e2]
min_error = float("inf")
C_ret = 0
gamma_ret = 0

for c in C:
	for g in gamma:
		# 1. SVR trained on train_x, train_y using C, g
		svr_rbf = SVR(kernel='rbf', C=c, gamma=g)
		# 2. tested on test_x, test_y
		svr_rbf.fit(train_x, train_y)
		#    1. predict on test_x using learned SVR to get pred_y
		pred_y = svr_rbf.predict(test_x)
		#    2. compute how different pred_y is from test_y (MSE)
		error = mse(pred_y, test_y)
		print "MSE for C=%d, gamma=%d: %f" %(c, g, error)
		#    3. remember C, g values for the learned SVR that performed best
		if (error < min_error):
			min_error = error
			C_ret = c
			gamma_ret = g

print "------------------------------------"
print "Best hyperparameter: C=%f, gamma=%f" %(C_ret, gamma_ret)

