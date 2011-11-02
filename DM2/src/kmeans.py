import numpy as np

from sklearn.metrics import euclidean_distances


def k_means(X, k, centers=None, max_iterations=300, ridge=1e-6, verbose=False):
    """
    Compute k-means !

    params
    -------
        X: array

        k: number of clusters
    """
    n_samples, n_features = X.shape

    # Initialisation - done randomly
    idxs = np.arange(n_samples)
    np.random.shuffle(idxs)
    seeds = idxs[:k]
    centers = X[seeds]

    for i in range(max_iterations):
        old_centers = centers.copy()
        distances = euclidean_distances(X, centers)
        labels = distances.argmin(axis=1)
        # compute the new centers
        for center_idx in range(k):
            mask = labels == center_idx

            centers[center_idx] = X[mask].sum(axis=0) / mask.sum(axis=0)

        if ((old_centers - centers)**2).sum() < ridge:
            if not verbose:
                print "Break out the loop at iteration %i" % i
            break

    # Now compute final labels
    distances = euclidean_distances(X, centers)
    labels = distances.argmin(axis=1)
    return centers, labels


def calculate_labels(centers, X):
    distances = euclidean_distances(X, centers)
    labels = distances.argmin(axis=1)
    return labels


class KMeans(object):
    def __init__(self, k=8):
        self.k = k


    def fit(self, X, centers=None, ridge=1e-6, verbose=False):
        self.centers_, labels = k_means(X, self.k,
                                        centers=centers,
                                        ridge=ridge,
                                        verbose=verbose)
        return labels

    def predict(self, X):
        return calculate_labels(self.centers_, X)





