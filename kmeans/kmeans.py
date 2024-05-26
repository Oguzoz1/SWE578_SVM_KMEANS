import numpy as np
import matplotlib.pyplot as plt
import os

class DataLoader:
    def __init__(self, data_file, labels_file):
        self.data_file = data_file
        self.labels_file = labels_file

    def load_data(self):
            data = np.load(self.data_file)
            labels = np.load(self.labels_file)
            return data, labels

class DataVisualizer:
    def plot_data_with_labels(self, data, labels):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = labels == label
            plt.scatter(data[label_indices, 0], data[label_indices, 1], label=f'Label {label}')
        plt.legend()
        plt.title('Data Visualization')
        plt.show()

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        rng = np.random.default_rng()
        # Randomly initialize centroids
        self.centroids = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = self._assign_clusters(X)

            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Check converge
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

class ClusteringVisualizer:
    def plot_clusters(self, data, labels, centroids):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            label_indices = labels == label
            #Scattering labels here
            plt.scatter(data[label_indices, 0], data[label_indices, 1], label=f'Cluster {label}')
        #Placing centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
        plt.legend()
        plt.title('Kmeans')
        plt.show()

class KMeansRun:
    def __init__(self, data_loader, data_visualizer, kmeans, clustering_visualizer):
        self.data_loader = data_loader
        self.data_visualizer = data_visualizer
        self.kmeans = kmeans
        self.clustering_visualizer = clustering_visualizer

    def run(self):
        data, labels = self.data_loader.load_data()
        self.data_visualizer.plot_data_with_labels(data, labels)
        
        self.kmeans.fit(data)
        self.clustering_visualizer.plot_clusters(data, self.kmeans.labels_, self.kmeans.centroids)

current_dir = os.path.dirname(__file__)
data_file = os.path.join(current_dir, 'data.npy')
labels_file = os.path.join(current_dir, 'label.npy')

data_loader = DataLoader(data_file, labels_file)
data_visualizer = DataVisualizer()
# You can change the number of clusters here
kmeans = KMeans(n_clusters=3)  
clustering_visualizer = ClusteringVisualizer()

KmeansRunner = KMeansRun(data_loader, data_visualizer, kmeans, clustering_visualizer)
KmeansRunner.run()
