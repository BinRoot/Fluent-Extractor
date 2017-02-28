import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)

xs_a_1 = np.random.rand(10, 1)
xs_a_2 = np.random.rand(10, 1)
xs_a = np.hstack((xs_a_1, xs_a_2))

xs_b_1 = 10 + np.random.rand(10, 1)
xs_b_2 = 5 + np.random.rand(10, 1)
xs_b = np.hstack((xs_b_1, xs_b_2))

X = np.vstack((xs_a, xs_b))

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
ps = gmm.predict(X)
ms = gmm.means_
cs = gmm.covariances_

print('ps', ps)
print('ms', ms)
print('cs', cs)
