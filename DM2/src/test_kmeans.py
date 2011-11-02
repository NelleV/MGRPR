"""
Test
"""

import numpy as np
import pylab as pl

from kmeans import KMeans, euclidean_distances

##############################################################################
# Generate sample data
np.random.seed(0)

centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)

std = 0.7
n_points_per_cluster = 300
X = np.empty((0, 2))
for i in range(n_clusters):
    X = np.r_[X, centers[i] + std * np.random.randn(n_points_per_cluster, 2)]
# Let's shuffle the data
np.random.shuffle(X) 

##############################################################################
# Compute clustering with Means

k_means = KMeans(k=3)
k_means_labels = k_means.fit(X.copy())
k_means_cluster_centers = k_means.centers_
k_means_labels_unique = np.unique(k_means_labels)

##############################################################################
# Plot result

fig = pl.figure()
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# That's a bit complicated... Because I stole that code from the scikit, where
# we wanted to do a comparison - I'm too lazy to delete the useless stuff

# KMeans
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
pl.text(-3.5, 2.7,  'train time: %.2fs')



