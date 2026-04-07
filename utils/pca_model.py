import numpy as np
from sklearn.decomposition import PCA

def fit_pca(X, n_components=3):
    """
    Fit PCA and return the model and transformed data.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def get_principal_vectors(pca):
    """
    Return the principal component vectors (directions).
    """
    return pca.components_

def get_variance_explained(pca):
    """
    Return the explained variance ratio for each component.
    """
    return pca.explained_variance_ratio_
