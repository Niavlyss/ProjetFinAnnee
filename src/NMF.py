import numpy as np
import matplotlib.image as mp
import numpy.ma as ma
import matplotlib.pyplot as plt
import random
import sklearn
from sklearn.cluster import KMeans

from scipy.sparse.linalg import svds



Mtest = np.ones((400,400))

n,m = Mtest.shape

for i in range(0,n):
	for j in range(0,m):
		Mtest [i][j] = random.choice([0,1])

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


def deepNMFTest(nbCouche, tailleCouche, M):
    n,m = M.shape


    for i in range(nbCouche):
        m = tailleCouche[i]
        U,V = appr_seminmf(M, m)

        fic1 = "U{}.csv".format(i)
        fic2 = "V{}.csv".format(i)
        np.savetxt(fic1,U, delimiter=",")
        np.savetxt(fic2,V, delimiter=",")

    return U,V


U,V = deepNMFTest(2,[300,100],Mtest)

