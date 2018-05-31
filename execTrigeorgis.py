from __future__ import print_function

import numpy as np
import sklearn

from sklearn.cluster import KMeans
from scipy.io import loadmat
from dsnmf import DSNMF
mat = loadmat('data/PIE_pose27.mat', struct_as_record=False, squeeze_me=True)

data, gnd = mat['fea'].astype('float32'), mat['gnd']

# Normalise each feature to have an l2-norm equal to one.
data /= np.linalg.norm(data, 2, 1)[:, None]

n_classes = np.unique(gnd).shape[0]
kmeans = KMeans(n_classes, precompute_distances=False)


def evaluate_nmi(X):
    pred = kmeans.fit_predict(X)
    score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)

    return score


dsnmf = DSNMF(data, layers=(400, 100))

for epoch in range(1000):
    residual = float(dsnmf.train_fun())

print("Epoch {}. Residual [{:.2f}]".format(epoch, residual), end="\r")

fea = dsnmf.get_features().T # this is the last layers features i.e. h_2
pred = kmeans.fit_predict(fea)
score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)

print("NMI: {:.2f}%".format(100 * score))