from sklearn.decomposition import NMF
import pandas
import numpy as np
import matplotlib.image as mp
import numpy.ma as ma
import matplotlib.pyplot as plt
import random


def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    
gray = mp.imread("../black.png")
gray = rgb2gray(gray)
msk = np.zeros(gray.size)

for i in range(0,len(msk)):
	msk[i] = random.choice([0, 1])
	


gray = ma.array(gray, mask = msk)

plt.imshow(gray,cmap = plt.get_cmap('gray'))
plt.show()
           
print(gray)
              
model = NMF(n_components=1, init='random', random_state=0, beta_loss='frobenius', max_iter=500, tol=0.0001)
W = model.fit_transform(gray)
H = model.components_


X1 = W.dot(H)
print(X1)

plt.imshow(X1,cmap = plt.get_cmap('gray'))
plt.show()

