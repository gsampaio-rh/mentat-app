import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot


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

    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg}")

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

    fig = plt.figure(figsize=(14, 10))
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
    save_plot(fig, "apply_pca.png")
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

    fig = plt.figure(figsize=(14, 10))
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
    save_plot(fig, "apply_tsne.png")
    plt.show()

    return tsne_df


def tune_kmeans_clustering(scaled_data, max_clusters=10):
    """
    Tune the number of clusters for K-Means clustering using silhouette scores.

    Args:
    - scaled_data (np.array): Scaled data.
    - max_clusters (int): Maximum number of clusters to test.

    Returns:
    - int: Best number of clusters.
    - float: Best silhouette score.
    - np.array: Cluster labels for the best clustering.
    - KMeans: Fitted KMeans model for the best clustering.
    """
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    best_num_clusters = 0
    best_silhouette_score = -1
    best_clusters = None
    best_kmeans_model = None

    for num_clusters in cluster_range:
        clusters, kmeans_model = apply_kmeans_clustering(
            scaled_data, num_clusters=num_clusters
        )
        score = silhouette_score(scaled_data, clusters)
        silhouette_scores.append(score)

        if score > best_silhouette_score:
            best_silhouette_score = score
            best_num_clusters = num_clusters
            best_clusters = clusters
            best_kmeans_model = kmeans_model

    # Validate the clustering
    print(
        f"Best number of clusters: {best_num_clusters} with silhouette score: {best_silhouette_score}"
    )

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=list(cluster_range), y=silhouette_scores, marker="o")
    plt.axvline(
        x=best_num_clusters,
        linestyle="--",
        color="r",
        label=f"Best: {best_num_clusters} clusters",
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for Different Numbers of Clusters")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    save_plot(plt.gcf(), "silhouette_scores.png")
    plt.show()

    return best_num_clusters, best_silhouette_score, best_clusters, best_kmeans_modelns.lineplot(x=list(cluster_range), y=silhouette_scores, marker="o")
