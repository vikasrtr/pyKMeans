"""
A working demo using KMeans

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('data.mat')
X = np.array(data['X'])

from KMeans import KMeans

k = 3
est = KMeans(k)

c = est.train(X)

colors=np.array(['green', 'red', 'blue'])
# lets plot on matplotlib
for i in range(k):
    x = X[np.where(c == i)[0]]
    plt.scatter(x[:, 0], x[:, 1], color=colors[i])

# plt.savefig('clustering_example.png')
plt.show()