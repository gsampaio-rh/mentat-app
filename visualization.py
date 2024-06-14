# visualization.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from utils import save_plot

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
    save_plot(fig, "summary_statistics.png")
    plt.show()


def plot_all_metrics_single_chart(df, features):
    """
    Plot all key metrics in a single chart with different colors and markers.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    """
    fig = plt.figure(figsize=(14, 8))
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
    save_plot(fig, "all_metrics_single_chart.png")
    plt.show()


def plot_bubble_chart(data):
    """
    Plot a bubble chart for the specified data.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    """
    fig = plt.figure(figsize=(14, 10))
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
    save_plot(fig, "bubble_chart.png")
    plt.show()


def plot_temporal_trends(data, metrics):
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data.set_index("Timestamp", inplace=True)
    fig = plt.figure(figsize=(14, 8))
    for metric in metrics:
        data[metric].resample("D").mean().plot(label=metric)
    plt.title("Temporal Trends of Business Metrics", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    save_plot(fig, "temporal_trends.png")
    plt.show()


def visualize_clusters(data, features, clusters):
    """
    Visualize clusters using scatter plots.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    - clusters (array): Array of cluster labels.
    """
    fig = plt.figure(figsize=(21, 7))
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
    save_plot(fig, "visualize_clusters.png")
    plt.show()


def plot_resource_utilization_efficiency(cluster_profiles):
    """
    Plot resource utilization efficiency by cluster.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    cluster_profiles[
        [
            "CPU Utilization (%)",
            "Memory Utilization (%)",
            "Network I/O Throughput (Mbps)",
            "Disk I/O Throughput (MB/s)",
        ]
    ].plot(kind="bar", ax=ax, alpha=0.75, colormap="viridis")
    ax.set_title(
        "Resource Utilization Efficiency by Cluster", fontsize=16, weight="bold"
    )
    ax.set_xlabel("Cluster", fontsize=14)
    ax.set_ylabel("Utilization", fontsize=14)
    ax.legend(title="Metrics")
    ax.grid(True, linestyle="--", linewidth=0.5)
    
    save_plot(fig, "resource_utilization_efficiency.png")
    plt.show()


def plot_cost_benefit_analysis(cluster_profiles):
    """
    Plot cost-benefit analysis by cluster.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.
    """
    fig=plt.figure(figsize=(14, 8))

    # Scatter plot for Operational Costs vs. Customer Satisfaction
    plt.scatter(
        cluster_profiles["Operational Costs ($)"],
        cluster_profiles["Customer Satisfaction (CSAT)"],
        s=200,
        alpha=0.75,
        label="CSAT vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    # Scatter plot for Operational Costs vs. Service Uptime
    plt.scatter(
        cluster_profiles["Operational Costs ($)"],
        cluster_profiles["Service Uptime (%)"],
        s=200,
        alpha=0.75,
        label="Uptime vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    plt.title("Cost-Benefit Analysis", fontsize=16, weight="bold")
    plt.xlabel("Operational Costs ($)", fontsize=14)
    plt.ylabel("Performance Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Save plot
    save_plot(fig, "cost_benefit_analysis.png")
    plt.show()


def plot_server_config_metrics(server_config_summary):
    """
    Plot performance metrics by server configuration.

    Args:
    - server_config_summary (pd.DataFrame): DataFrame containing the server configuration summary.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    server_config_summary[
        [
            "CPU Utilization (%)",
            "Memory Utilization (%)",
            "Network I/O Throughput (Mbps)",
            "Disk I/O Throughput (MB/s)",
        ]
    ].plot(kind="bar", ax=ax, alpha=0.75, colormap="viridis")
    ax.set_title(
        "Average Performance Metrics by Server Configuration",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel("Server Configuration", fontsize=14)
    ax.set_ylabel("Average Metrics", fontsize=14)
    ax.legend(title="Metrics")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Save plot
    save_plot(fig, "server_config_metrics.png")
    plt.show()


def plot_cost_vs_performance(server_config_summary):
    """
    Plot cost vs performance metrics for each server configuration.

    Args:
    - server_config_summary (pd.DataFrame): DataFrame containing the server configuration summary.
    """
    fig=plt.figure(figsize=(14, 8))

    # Scatter plot for Operational Costs vs. Customer Satisfaction
    plt.scatter(
        server_config_summary["Operational Costs ($)"],
        server_config_summary["Customer Satisfaction (CSAT)"],
        s=200,
        alpha=0.75,
        label="CSAT vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    # Scatter plot for Operational Costs vs. Service Uptime
    plt.scatter(
        server_config_summary["Operational Costs ($)"],
        server_config_summary["Service Uptime (%)"],
        s=200,
        alpha=0.75,
        label="Uptime vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    plt.title(
        "Cost vs Performance Metrics by Server Configuration",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel("Operational Costs ($)", fontsize=14)
    plt.ylabel("Performance Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)

    # Save plot
    save_plot(fig,"cost_vs_performance.png")
    plt.show()
