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


def _e_step_1(X, centers, p, pi):
    """
    Calculate the M step

    """
    mindist = np.empty(X.shape[0])
    mindist.fill(np.infty)

    for center_idx in range(centers.shape[0]):
        distances = np.sum((X - centers[centers_idx]) ** 2, axis=1)
        labels[dist < mindist] = center_idx
        mindst = np.minimum(dist, mindist)

    return labels


if __name__ == "__main__":
    # Generate sample data
#    np.random.seed(0)
#
#    centers = [[1, 1], [-1, -1], [1, -1]]
#    n_clusters = len(centers)
#
#    std = 0.7
#    n_points_per_cluster = 300
#    X = np.empty((0, 2))
#
#    for i in range(n_clusters):
#        X = np.r_[X, centers[i] + std * np.random.randn(n_points_per_cluster, 2)]
#    # Let's shuffle the data
#    np.random.shuffle(X)


    n_clusters = 4
    X = utils.load_data('EMGaussienne.data')

    max_iterations = 150
    ridge = 1e-6
    verbose = True

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



    # Let's plot the results

    import pylab as pl

    x = np.linspace(X.T[0, :].min(), X.T[0, :].max(), num=100)
    y = np.linspace(X.T[1, :].min(), X.T[1, :].max(), num=100)
    x, y = np.meshgrid(x, y)
    xx = np.c_[x.ravel(), y.ravel()]

    fig = pl.figure()
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#00465F']
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(n_clusters), colors):

        my_members = k_means_labels == k
        cluster_center = mu[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=6)

    for k in range(n_clusters):
        z = _calculate_normal(xx, mu[k, :], sigma[k, :])
        z = z.reshape(x.shape)
        pl.contour(x, y, z)

    ax.set_title('Algorithme EM')

