"""
A basic implementation of K-Means clustering algorithm

"""

import numpy as np


class KMeans(object):

    """KMeans implementation.
    Requires K for initialisation"""

    def __init__(self, k, max_iter=20):
        self._k = k
        self._max_iter = max_iter

    def train(self, X):
        """Performs clustering

        Parameters:
        X : Numpy Array of form [num_samples, dimension]
        """

        # Generate K random centroids
        random_nos = np.random.uniform(0, X.shape[0] - 1, self._k).astype(int)
        centroids = X[random_nos]
        old_centroids = np.zeros(centroids.shape)

        # create a vector of labels (0..k-1 for k-clusters)
        c = np.zeros(shape=(X.shape[0], 1))

        iters = 1
        while not self._should_stop(iters, old_centroids, centroids):
            iters += 1
            # loop through all points - NP-Hard !!!
            for i in range(X.shape[0]):
                old_dist = 99999
                for j in range(0, self._k):
                    dist = np.linalg.norm(X[i] - centroids[j])
                    if dist < old_dist:
                        # assign the jth cluster
                        old_dist = dist
                        c[i] = j

            # Update centroid to new values
            for i in range(self._k):
                # get all point with label i
                label_j = X[np.where(c == i)[0]]
                centroids[i] = np.sum(label_j, axis=0) / label_j.shape[0]

        # return all new labels for each point
        return c

    def _should_stop(self, iters, old_centroids, centroids):
        if iters > self._max_iter:
            return True
        return np.array_equal(old_centroids, centroids)
