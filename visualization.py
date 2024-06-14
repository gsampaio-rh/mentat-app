# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull


def plot_summary_statistics(data, features):
    """
    Plot summary statistics for the specified features.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot summary statistics for.
    """
    summary = data[features].describe().T
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    plt.title("Summary Statistics of Server Metrics", fontsize=16, weight="bold")
    plt.show()


def plot_all_metrics_single_chart(df, features):
    """
    Plot all key metrics in a single chart with different colors and markers.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    """
    plt.figure(figsize=(14, 8))
    colors = ["red", "green", "blue", "purple"]
    markers = ["o", "s", "D", "X"]

    for i, metric in enumerate(features):
        plt.scatter(
            df.index, df[metric], label=metric, color=colors[i], marker=markers[i], s=10
        )

    plt.title("All Key Metrics (Normalized)", fontsize=16, weight="bold")
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Normalized Value", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_bubble_chart(data):
    """
    Plot a bubble chart for the specified data.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    """
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        data["CPU Utilization (%)"],
        data["Memory Utilization (%)"],
        s=data["Network I/O Throughput (Mbps)"] / 500,
        c=data["Disk I/O Throughput (MB/s)"],
        alpha=0.6,
        cmap="viridis",
        edgecolor="w",
        linewidth=0.5,
    )
    plt.colorbar(scatter, label="Disk I/O Throughput (MB/s)")
    plt.xlabel("CPU Utilization (%)", fontsize=14)
    plt.ylabel("Memory Utilization (%)", fontsize=14)
    plt.title("Bubble Chart of Chaotic Server Metrics", fontsize=16, weight="bold")
    plt.xlim(20, 120)
    plt.ylim(10, 110)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_temporal_trends(data, metrics):
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data.set_index("Timestamp", inplace=True)
    plt.figure(figsize=(14, 8))
    for metric in metrics:
        data[metric].resample("D").mean().plot(label=metric)
    plt.title("Temporal Trends of Business Metrics", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def visualize_clusters(data, features, clusters):
    """
    Visualize clusters using scatter plots.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    - clusters (array): Array of cluster labels.
    """
    plt.figure(figsize=(21, 7))
    colors = sns.color_palette("Set2", len(np.unique(clusters)))
    sns.set(style="whitegrid")

    plt.subplot(1, 3, 1)
    scatter = plt.scatter(
        data[features[0]],
        data[features[1]],
        s=data[features[2]] / 500,
        c=data[features[3]],
        alpha=0.7,
        cmap="viridis",
        edgecolor="w",
        linewidth=0.5,
    )
    plt.colorbar(scatter, label=features[3])
    plt.xlabel(features[0], fontsize=14)
    plt.ylabel(features[1], fontsize=14)
    plt.title("Unlabeled Data", fontsize=16, weight="bold")
    plt.grid(True)
    plt.legend(
        ["Size: Network I/O Throughput", "Color: Disk I/O Throughput"],
        loc="upper right",
    )

    plt.subplot(1, 3, 2)
    for i, cluster in enumerate(np.unique(clusters)):
        cluster_data = data[clusters == cluster]
        plt.scatter(
            cluster_data[features[0]],
            cluster_data[features[1]],
            c=[colors[i]],
            alpha=0.7,
            label=f"Cluster {cluster}",
            edgecolors="w",
            linewidth=0.5,
        )

    plt.xlabel(features[0], fontsize=14)
    plt.ylabel(features[1], fontsize=14)
    plt.title("Labeled Clusters", fontsize=16, weight="bold")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    for i, cluster in enumerate(np.unique(clusters)):
        cluster_data = data[clusters == cluster]
        plt.scatter(
            cluster_data[features[0]],
            cluster_data[features[1]],
            c=[colors[i]],
            alpha=0.7,
            label=f"Cluster {cluster}",
            edgecolors="w",
            linewidth=0.5,
        )
        if len(cluster_data) > 2:
            hull = ConvexHull(cluster_data[features[:2]])
            hull_points = np.append(hull.vertices, hull.vertices[0])
            plt.plot(
                cluster_data[features[0]].values[hull_points],
                cluster_data[features[1]].values[hull_points],
                c=colors[i],
                alpha=0.7,
                linewidth=2,
            )
            plt.fill(
                cluster_data[features[0]].values[hull_points],
                cluster_data[features[1]].values[hull_points],
                c=colors[i],
                alpha=0.2,
            )

    plt.xlabel(features[0], fontsize=14)
    plt.ylabel(features[1], fontsize=14)
    plt.title("Cluster Plot with Boundaries", fontsize=16, weight="bold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
