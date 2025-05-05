import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage 

iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X_scaled)

silhouette_avg = silhouette_score(X_scaled, labels)

print(f"Number of clusters: 3")
print(f"Cluster Labels for first 10 samples:\n{labels[:10]}")
print(f"\nSilhouette Score: {silhouette_avg:.4f}")

linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()