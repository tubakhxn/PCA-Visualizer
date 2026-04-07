import numpy as np
from sklearn.datasets import make_blobs

def generate_high_dimensional_data(n_samples=300, n_features=5, n_clusters=3, cluster_std=1.5, random_state=42):
    """
    Generate synthetic high-dimensional Gaussian clusters.
    Returns X (data), y (labels)
    """
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                      cluster_std=cluster_std, random_state=random_state)
    return X, y
