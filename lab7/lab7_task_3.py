import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Завантаження даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини смуги пропускання
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Ініціалізація та навчання моделі MeanShift
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Центри кластерів
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters\n', cluster_centers)

# Отримання міток кластерів та кількості кластерів
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nNumber of clusters in input data =\n', num_clusters)

# Візуалізація
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), cycle(markers)):
    # Відображення точок, що належать кластеру
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, label=f"Cluster {i+1}")

# Відображення центрів кластерів
for center in cluster_centers:
    plt.plot(center[0], center[1], marker='o', markerfacecolor='red', 
             markeredgecolor='black', markersize=15)

plt.title('Clusters')
plt.legend()
plt.show()
