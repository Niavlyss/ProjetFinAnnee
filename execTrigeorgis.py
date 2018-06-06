from __future__ import print_function

import time

import numpy as np
import sklearn
from pandas import read_csv
from scipy.io import loadmat
from sklearn.cluster import KMeans

from src.code_Trigeorgis import DSNMF

mat = loadmat('data/PIE_pose27.mat', struct_as_record=False, squeeze_me=True)
# print(mat)
# data, gnd = mat['fea'].astype('float32'), mat['gnd']

df = read_csv('data/Dk1+Dk2_data_reduit_sans_outliers.csv', sep=";", skiprows=1)

data = df.values.astype('float32')
gnd = data


data = data.transpose()
# Normalise each feature to have an l2-norm equal to one.
data /= np.linalg.norm(gnd, 2, 1)[None, :]

n_classes = data.shape[0]
print(n_classes)
kmeans = KMeans(n_classes, precompute_distances=False)


dsnmf = DSNMF(data, layers=(22, 10 , 5))
tmps1 = time.time()
for epoch in range(10000):
    residual = float(dsnmf.train_fun())

    print("Epoch {}. Residual [{:.2f}]".format(epoch, residual), end="\r")

fea = dsnmf.get_features().T # this is the last layers features i.e. h_2
pred = kmeans.fit_predict(fea)

score = sklearn.metrics.normalized_mutual_info_score(fea[:,0], pred)

print("NMI: {:.2f}%".format(100 * score))

tmps2 = time.time()-tmps1
print("temps d'execution : %f" %tmps2)