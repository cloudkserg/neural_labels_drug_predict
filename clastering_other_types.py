from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels_, cmap='viridis')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('DBSCAN Clustering')
plt.show()

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply bisecting k-means
k = 2
kmeans = KMeans(n_clusters=k)
labels = [-1] * len(X)
clusters = [X]

while len(clusters) < k:
    best_cluster = None
    best_cost = None
    best_labels = None
    
    for i, cluster in enumerate(clusters):
        kmeans.fit(cluster)
        cost = sum([min([((kmeans.cluster_centers_[j][0] - point[0]) ** 2 + 
                          (kmeans.cluster_centers_[j][1] - point[1]) ** 2) for j in range(len(kmeans.cluster_centers_))]) 
                    for point in cluster])
        if best_cost is None or cost > best_cost:
            best_cluster = i
            best_cost = cost
            best_labels = kmeans.predict(cluster)
    
    new_cluster1 = clusters[best_cluster][best_labels == 0]
    new_cluster2 = clusters[best_cluster][best_labels == 1]
    clusters.pop(best_cluster)
    clusters.append(new_cluster1)
    clusters.append(new_cluster2)
    
    for i in range(len(labels)):
        if labels[i] == best_cluster:
            if best_labels[i] == 0:
                labels[i] = len(clusters) - 2
            else:
                labels[i] = len(clusters) - 1

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Bisecting K-Means Clustering')
plt.show()
