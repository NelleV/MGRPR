import pylab as pl

from utils import load_data
from kmeans import KMeans

X = load_data('EMGaussienne.data')
X_test = load_data('EMGaussienne.data')

n_clusters = 4
num_init = 3

##############################################################################
# Plot result

fig = pl.figure()
colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#00465F']

for ini in range(num_init):
    km = KMeans(k=n_clusters)

    k_means_labels, k_means_inertia = km.fit(X)
    k_means_cluster_centers = km.centers_
    k_means_labels_test, k_means_inertia_test = km.predict(X_test)


    # KMeans
    ax = fig.add_subplot(num_init, 2, 2 * ini + 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=6)
    ax.set_title('KMeans - inertia %d' % k_means_inertia)

    # KMeans
    ax = fig.add_subplot(num_init, 2, 2 * ini + 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels_test == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=6)
    ax.set_title('KMeans -- Test Data - inertia %d' % k_means_inertia_test)



pl.show()



