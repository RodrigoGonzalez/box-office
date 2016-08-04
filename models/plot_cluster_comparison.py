print(__doc__)

import time
import pandas as pd
import numpy as np

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# actor: 1, producer: 3, writer: 4, director: 8
df_actor = pd.read_csv("../html_postgres/person_clusters_1.csv")
df_producer = pd.read_csv("../html_postgres/person_clusters_3.csv")
df_writer = pd.read_csv("../html_postgres/person_clusters_4.csv")
df_director = pd.read_csv("../html_postgres/person_clusters_8.csv")

columns = ['total_box_office_revenues', 'total_lifetime_earnings', 'gender']

actors = df_actor[columns].get_values(), df_actor['name_id'].get_values()
producers = df_producer[columns].get_values(), df_producer['name_id'].get_values()
writers = df_writer[columns].get_values(), df_writer['name_id'].get_values()
directors = df_director[columns].get_values(), df_director['name_id'].get_values()

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

#

clustering_names = [ 'MiniBatchKMeans',
    'AffinityPropagation',
    'MeanShift',
    'SpectralClustering',
    'Ward',
    'AgglomerativeClustering',
    'DBSCAN',
    'Birch']

plt.figure(figsize=(len(clustering_names) * 4 + 6, 15))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.05)

plot_num = 1

count = 0

datasets = [actors, producers, writers, directors]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)

    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)

    birch = cluster.Birch(n_clusters=2)
    clustering_algorithms = [
        two_means,
        affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        a = pd.DataFrame(y_pred)

        print name
        count += 1
        print count
        print a

        a.to_csv('../html_postgres/person_clusters_labels_{}_{}.csv'.format(name, count), mode = 'w', index=False)

        # plot
        plt.subplot(4, len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
            plt.xlabel('life time earnings', fontsize=8)
            plt.ylabel('gender', fontsize=8)
        plt.scatter(X[:, 1], X[:, 2], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
        plot_num += 1

plt.savefig("clustering_2.png") # save as png


plt.show()
