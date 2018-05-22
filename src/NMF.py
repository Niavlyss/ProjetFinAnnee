from sklearn.decomposition import NMF
import pandas
import numpy as np
import matplotlib.image as mp
import numpy
import matplotlib.pyplot as plt


 
X = mp.imread("../nb.png")

model = NMF(n_components=4, init='nndsvd', random_state=0, beta_loss='frobenius', max_iter=5000)
W = model.fit_transform(X)
H = model.components_

print(W)
print(H)

X1 = W.dot(H)

plt.imshow(X1)
plt.show()

