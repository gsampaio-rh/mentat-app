import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import mpld3

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data, data_scaled


def perform_clustering(data_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_scaled)
    silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.2f}")
    return kmeans


def perform_pca(data_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    return data_pca


def detect_anomalies(data_scaled, contamination=0.1):
    iforest = IsolationForest(contamination=contamination)
    iforest.fit(data_scaled)
    anomalies = iforest.predict(data_scaled)
    return anomalies


def visualize_clusters(data_pca, kmeans_labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=kmeans_labels,
        cmap="viridis",
        marker="o",
        s=50,
        edgecolors="k",
    )
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.title("K-Means Clustering with PCA", fontsize=16)
    plt.legend(["Cluster 1", "Cluster 2", "Cluster 3"], fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()


def visualize_anomalies(data_pca, anomalies):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=anomalies,
        cmap="viridis",
        marker="o",
        s=50,
        edgecolors="k",
    )
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.title("Anomaly Detection using Isolation Forest", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()


def main():
    data, data_scaled = load_and_preprocess_data("data/insights_metrics.csv")
    kmeans = perform_clustering(data_scaled)
    data_pca = perform_pca(data_scaled)
    visualize_clusters(data_pca, kmeans.labels_)
    anomalies = detect_anomalies(data_scaled)
    visualize_anomalies(data_pca, anomalies)


if __name__ == "__main__":
    main()
