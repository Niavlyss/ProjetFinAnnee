from sklearn.decomposition import NMF
import pandas
import numpy as np
import matplotlib.image as mp
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T

from scipy.sparse.linalg import svds


def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    
gray = mp.imread("../black.png")
gray = rgb2gray(gray)
	       

def appr_seminmf(M, r):
    """
        Approximate Semi-NMF factorisation. 
        
        Parameters
        ----------
        M: array-like, shape=(n_features, n_samples)
        r: number of components to keep during factorisation
    """
    
    if r < 2:
        raise ValueError("The number of components (r) has to be >=2.")

    A, S, B = svds(M, r-1)
    S = np.diag(S)
    A = np.dot(A, S)
 
    m, n = M.shape
 
    for i in range(r-1):
        if B[i, :].min() < (-B[i, :]).min():
            B[i, :] = -B[i, :]
            A[:, i] = -A[:, i]
            
            
    if r == 2:
        U = np.concatenate([A, -A], axis=1)
    else:
        An = -np.sum(A, 1).reshape(A.shape[0], 1)
        U = np.concatenate([A, An], 1)
    
    V = np.concatenate([B, np.zeros((1, n))], 0)

    if r>=3:
        V -= np.minimum(0, B.min(0))
    else:
        V -= np.minimum(0, B)

    return U, V

n,m =gray.shape


for i in range(2):
	appr_seminmf(gray,m)
	m=m/2
	print(gray)
