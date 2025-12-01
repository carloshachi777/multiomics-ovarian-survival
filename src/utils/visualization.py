import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

try:
    import umap
except ImportError:
    umap = None

from sklearn.manifold import TSNE


def plot_latent_umap(
    Z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "UMAP of latent space",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
):
    """
    UMAP visualization of latent space Z (n_samples, latent_dim).
    """
    if umap is None:
        raise ImportError("Please install umap-learn to use UMAP visualization.")

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    Z_2d = reducer.fit_transform(Z)

    plt.figure(figsize=(6, 5))
    if labels is not None:
        scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, alpha=0.8)
        plt.legend(*scatter.legend_elements(), title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], alpha=0.8)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()


def plot_latent_tsne(
    Z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "t-SNE of latent space",
):
    """
    t-SNE visualization of latent space Z.
    """
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    Z_2d = tsne.fit_transform(Z)

    plt.figure(figsize=(6, 5))
    if labels is not None:
        scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels, alpha=0.8)
        plt.legend(*scatter.legend_elements(), title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.scatter(Z_2d[:, 0], Z_2d[:, 1], alpha=0.8)

    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.show()
