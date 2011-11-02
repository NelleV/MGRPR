import pylab as pl

from utils import load_data
from kmeans import KMeans

X = load_data('EMGaussienne.data')
X_test = load_data('EMGaussienne.data')

n_clusters = 4

km = KMeans(k=n_clusters)

k_means_labels = km.fit(X)
k_means_cluster_centers = km.centers_
k_means_labels_test = km.predict(X_test)

##############################################################################
# Plot result

fig = pl.figure()
colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#00465F']

# That's a bit complicated... Because I stole that code from the scikit, where
# we wanted to do a comparison - I'm too lazy to delete the useless stuff

# KMeans
ax = fig.add_subplot(1, 2, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=6)
ax.set_title('KMeans')

# KMeans
ax = fig.add_subplot(1, 2, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels_test == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=6)
ax.set_title('KMeans -- Test Data')



pl.show()



