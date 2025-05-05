import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

silhouette_avg = silhouette_score(X_scaled, labels)

print(f"Number of clusters: 3")
print(f"Cluster Centroids:\n{centroids}")
print(f"\nCluster Labels for first 10 samples:\n{labels[:10]}")
print(f"\nSilhouette Score: {silhouette_avg:.4f}")