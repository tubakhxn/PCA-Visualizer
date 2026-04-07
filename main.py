import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from utils.data_generator import generate_high_dimensional_data
from utils.pca_model import fit_pca, get_principal_vectors, get_variance_explained

# Generate synthetic data
X, y = generate_high_dimensional_data(n_samples=300, n_features=5, n_clusters=3, cluster_std=1.5, random_state=42)

# Fit PCA for 3D and 2D
pca_3d, X_3d = fit_pca(X, n_components=3)
pca_2d, X_2d = fit_pca(X, n_components=2)

# Get principal vectors and explained variance
vectors_3d = get_principal_vectors(pca_3d)
variance_3d = get_variance_explained(pca_3d)
vectors_2d = get_principal_vectors(pca_2d)
variance_2d = get_variance_explained(pca_2d)

# Project original data to 3D for visualization
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', s=40, alpha=0.7)
ax.set_title('3D PCA Projection')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Draw principal component vectors as arrows
mean_3d = np.mean(X_3d, axis=0)
for i in range(3):
    vec = vectors_3d[i] * 3 * np.sqrt(variance_3d[i])  # scale for visibility
    ax.quiver(mean_3d[0], mean_3d[1], mean_3d[2], vec[0], vec[1], vec[2], color='r', linewidth=2)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Cluster Label')

# 2D PCA plot
ax2 = fig.add_subplot(122)
scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', s=40, alpha=0.7)
ax2.set_title('2D PCA Projection')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

# Draw principal component vectors in 2D
mean_2d = np.mean(X_2d, axis=0)
for i in range(2):
    vec = vectors_2d[i] * 3 * np.sqrt(variance_2d[i])
    ax2.arrow(mean_2d[0], mean_2d[1], vec[0], vec[1], color='r', width=0.05, head_width=0.2, length_includes_head=True)

# Add colorbar
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Cluster Label')

# Show explained variance
explained = f"Explained variance (3D): {variance_3d.round(3)}\nTotal: {variance_3d.sum():.2f}\n"
explained += f"Explained variance (2D): {variance_2d.round(3)}\nTotal: {variance_2d.sum():.2f}"
fig.text(0.5, 0.01, explained, ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.show()
