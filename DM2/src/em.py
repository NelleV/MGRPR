import numpy as np

from kmeans import KMeans
import utils

def _calculate_normal(X, mu, sigma):
    """
    Calculate the probability p of X following normal law
    """
    a = 1. / (2 * np.pi * np.linalg.det(sigma))
    S_inv = np.linalg.inv(sigma) 
    b = np.diag(np.dot(np.dot((X - mu), S_inv), (X - mu).T))
    p = a * np.exp( -1./2 * b)
    return p


def calculate_em(X, n_clusters,
                 diag=False, ridge=1e-6, verbose=False, max_iterations=100):
    n_samples, n_features = X.shape
    # Initialise the data using kmeans
    k_means = KMeans(k=n_clusters)
    k_means_labels = k_means.fit(X.copy())
    k_means_cluster_centers = k_means.centers_

    # OK, so we've got the centers and the labels. Let's now compute the EM
    # algorithm
    tau = np.zeros((n_samples, n_clusters))
    mu = np.zeros((n_clusters, n_features))
    sigma = np.zeros((n_clusters, n_features, n_features))
    p = np.zeros((n_clusters, n_samples))
    # FIXME shouldbe able to do the following using pure matric arithmetics
    for i, element in enumerate(k_means_labels):
        tau[i, element] = 1

    for j in range(max_iterations):
        old_mu = mu.copy()
        for i in range(n_clusters):
            mu[i] = (tau[:, i].reshape((tau.shape[0], 1)) * X).sum(axis=0) / \
                (tau[:, i]).sum()

        for i in range(n_clusters):
            a = 0
            for n in range(n_samples):
                b = (X[n, :] - mu[i]).reshape((2, 1))
                a += tau[n, i] * np.dot(b, b.T)
            if diag:
                sigma[i, :] = a.mean() / tau[:, i].sum() * np.identity(mu.shape[1])
            else:
                sigma[i, :] = a / tau[:, i].sum()


        tpi = tau.sum(axis=1) / n_samples
        for i in range(n_clusters):
            p[i, :] = _calculate_normal(X, mu[i, :], sigma[i, :])

        for i in range(n_clusters):
            tau.T[i, :] = tpi[i] * p[i, :] / (tpi * p).sum(axis=0)

        if ((old_mu - mu)**2).sum() < ridge:
            if verbose:
                print "break at iterations %d" % j
            break

    return mu, sigma



