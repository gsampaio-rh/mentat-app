# clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def apply_kmeans_clustering(scaled_data, num_clusters=3):
    """
    Apply K-Means clustering to the scaled data.

    Args:
    - scaled_data (np.array): Scaled data.
    - num_clusters (int): Number of clusters.

    Returns:
    - np.array: Cluster labels.
    - KMeans: Fitted KMeans model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans


def apply_pca(scaled_data, clusters):
    """
    Apply PCA to the scaled data and plot the results.

    Args:
    - scaled_data (np.array): Scaled data.
    - clusters (np.array): Cluster labels.

    Returns:
    - pd.DataFrame: DataFrame containing PCA results and cluster labels.
    - PCA: Fitted PCA model.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters

    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")
    custom_palette = sns.color_palette("Set2", len(np.unique(clusters)))
    scatter = sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=custom_palette,
        data=pca_df,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
    )

    legend = scatter.legend(
        title="Cluster",
        title_fontsize="13",
        fontsize="11",
        loc="upper right",
        frameon=True,
        shadow=True,
        borderpad=1,
    )

    plt.title("PCA of Server Metrics with Clusters", fontsize=16, weight="bold")
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

    return pca_df, pca


def get_pca_loadings(pca, features):
    """
    Get PCA loadings for the specified features.

    Args:
    - pca (PCA): Fitted PCA model.
    - features (list): List of features.

    Returns:
    - pd.DataFrame: DataFrame containing PCA loadings.
    """
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
    return loadings


def apply_tsne(scaled_data, clusters):
    """
    Apply t-SNE to the scaled data and plot the results.

    Args:
    - scaled_data (np.array): Scaled data.
    - clusters (np.array): Cluster labels.

    Returns:
    - pd.DataFrame: DataFrame containing t-SNE results and cluster labels.
    """
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=X_tsne, columns=["TSNE1", "TSNE2"])
    tsne_df["Cluster"] = clusters

    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")
    custom_palette = sns.color_palette("Set2", len(np.unique(clusters)))
    sns.scatterplot(
        x="TSNE1",
        y="TSNE2",
        hue="Cluster",
        palette=custom_palette,
        data=tsne_df,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
    )

    plt.title("t-SNE of Server Metrics with Clusters", fontsize=16, weight="bold")
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(
        title="Cluster",
        title_fontsize="13",
        fontsize="11",
        loc="upper right",
        frameon=True,
        shadow=True,
        borderpad=1,
    )
    plt.show()

    return tsne_df
