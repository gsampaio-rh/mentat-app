import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest


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


def highlight_anomalies_in_data(data, anomalies):
    data["Anomaly"] = anomalies
    anomalies_data = data[data["Anomaly"] == -1]
    normal_data = data[data["Anomaly"] == 1]

    # Plotting the correlation heatmap to identify important relationships
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Metrics")
    plt.show()

    # Highlight metrics with significant correlation to anomalies
    anomaly_correlation = (
        correlation_matrix["Anomaly"].drop("Anomaly").sort_values(ascending=False)
    )
    print("Metrics with highest correlation to anomalies:")
    print(anomaly_correlation)

    key_metrics = anomaly_correlation.head(
        3
    ).index  # Top 3 metrics correlated with anomalies

    # Generate scatter plots for the top 3 metrics
    plt.figure(figsize=(14, 7))
    symbols = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]
    for i, column in enumerate(key_metrics):
        plt.scatter(
            normal_data.index,
            normal_data[column],
            label=column,
            alpha=0.5,
            marker=symbols[i % len(symbols)],
        )
        plt.scatter(
            anomalies_data.index,
            anomalies_data[column],
            label=f"Anomaly in {column}",
            edgecolors="r",
            marker=symbols[i % len(symbols)],
        )

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.title("Detected Anomalies in Key Metrics")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.show()

    # Bar chart for all correlation values
    plt.figure(figsize=(12, 6))
    anomaly_correlation.plot(kind="bar", color="skyblue")
    plt.title("Correlation of Metrics with Anomalies")
    plt.xlabel("Metrics")
    plt.ylabel("Correlation with Anomalies")
    plt.xticks(rotation=45, ha="right")
    plt.show()

    # Print summary of anomalies
    print("Summary of Anomalies Detected:")
    print(anomalies_data.describe())


def main():

    file_path = "data/insights_metrics.csv"
    data, data_scaled = load_and_preprocess_data(file_path)
    kmeans = perform_clustering(data_scaled)
    data_pca = perform_pca(data_scaled)
    visualize_clusters(data_pca, kmeans.labels_)
    anomalies = detect_anomalies(data_scaled)
    highlight_anomalies_in_data(data, anomalies)


if __name__ == "__main__":
    main()
