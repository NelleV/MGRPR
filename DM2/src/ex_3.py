import numpy as np
import pylab as pl
import utils
from em import calculate_em, _calculate_normal
from kmeans import KMeans


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

mu, sigma = calculate_em(X, n_clusters)

# Let's plot the results

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


