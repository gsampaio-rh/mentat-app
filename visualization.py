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
    num_features = len(features)
    num_cols = 2  # Number of columns
    num_rows = (num_features + 1) // num_cols  # Calculate number of rows

    # Define colors for each feature
    colors = sns.color_palette("husl", num_features)

    # Adjust the figure size to fit a MacBook 15" screen
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))

    for i, feature in enumerate(features):
        row, col = divmod(i, num_cols)
        sns.histplot(data[feature], kde=True, ax=axes[row, col], color=colors[i])
        axes[row, col].set_title(f"{feature} Distribution")
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Count")
        axes[row, col].legend([feature])

    # Remove any empty subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flat[j])

    # Adjust layout
    plt.tight_layout()
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
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    for i, metric in enumerate(metrics):
        data.plot(x="Timestamp", y=metric, ax=axes[i])
        axes[i].set_title(f"Temporal Trend of {metric}")
    plt.tight_layout()
    save_plot(fig, "temporal_trends.png")
    plt.show()


def visualize_clusters(data, cluster_labels, features):
    sns.pairplot(data[features + [cluster_labels]], hue=cluster_labels)
    plt.show()


def visualize_all_labelled_non_labelled_clusters(data, features, clusters):
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


def calculate_thresholds(data, features):
    thresholds = {}
    for feature in features:
        thresholds[feature] = {
            "low": data[feature].quantile(0.25),
            "high": data[feature].quantile(0.75),
        }
    return thresholds


def plot_shaded_regions(thresholds):
    plt.axvspan(
        0,
        thresholds["CPU Utilization (%)"]["low"],
        color="green",
        alpha=0.2,
        label="Low CPU Utilization",
    )
    plt.axvspan(
        thresholds["CPU Utilization (%)"]["low"],
        thresholds["CPU Utilization (%)"]["high"],
        color="blue",
        alpha=0.1,
        label="Moderate CPU Utilization",
    )
    plt.axvspan(
        thresholds["CPU Utilization (%)"]["high"],
        1,
        color="red",
        alpha=0.2,
        label="High CPU Utilization",
    )

    plt.axhspan(
        0,
        thresholds["Memory Utilization (%)"]["low"],
        color="green",
        alpha=0.2,
        label="Low Memory Utilization",
    )
    plt.axhspan(
        thresholds["Memory Utilization (%)"]["low"],
        thresholds["Memory Utilization (%)"]["high"],
        color="blue",
        alpha=0.1,
        label="Moderate Memory Utilization",
    )
    plt.axhspan(
        thresholds["Memory Utilization (%)"]["high"],
        1,
        color="red",
        alpha=0.2,
        label="High Memory Utilization",
    )


def plot_resource_utilization_efficiency(avg_utilization_df):
    # Define thresholds for high and low utilization
    high_utilization_threshold = avg_utilization_df[['CPU Utilization (%)', 'Memory Utilization (%)']].quantile(0.75)
    low_utilization_threshold = avg_utilization_df[['CPU Utilization (%)', 'Memory Utilization (%)']].quantile(0.25)

    # Assign colors based on utilization
    avg_utilization_df['Color'] = 'blue'  # Default color
    avg_utilization_df.loc[(avg_utilization_df['CPU Utilization (%)'] >= high_utilization_threshold['CPU Utilization (%)']) &
                           (avg_utilization_df['Memory Utilization (%)'] >= high_utilization_threshold['Memory Utilization (%)']), 'Color'] = 'red'
    avg_utilization_df.loc[(avg_utilization_df['CPU Utilization (%)'] <= low_utilization_threshold['CPU Utilization (%)']) &
                           (avg_utilization_df['Memory Utilization (%)'] <= low_utilization_threshold['Memory Utilization (%)']), 'Color'] = 'green'

    # Plot the data
    fig = plt.figure(figsize=(10, 6))
    for i in range(len(avg_utilization_df)):
        plt.scatter(avg_utilization_df['CPU Utilization (%)'][i], avg_utilization_df['Memory Utilization (%)'][i], color=avg_utilization_df['Color'][i])
        plt.text(avg_utilization_df['CPU Utilization (%)'][i], avg_utilization_df['Memory Utilization (%)'][i], 
                 avg_utilization_df['Server Configuration'][i], fontsize=9)

    # Add legends for better and worse performance
    high_label = 'High Utilization (Red)'
    low_label = 'Low Utilization (Green)'
    plt.scatter([], [], color='red', label=high_label)
    plt.scatter([], [], color='green', label=low_label)
    plt.legend(loc='upper right')

    plt.title('Resource Utilization Efficiency')
    plt.xlabel('Average CPU Utilization (%)')
    plt.ylabel('Average Memory Utilization (%)')
    plt.grid(True)
    save_plot(fig, "resource_utilization_efficiency.png")
    plt.show()


def calculate_cost_reduction_thresholds(data):
    thresholds = {
        "high_cost": data["Operational Costs ($)"].quantile(0.75),
        "low_impact_cs": data["Customer Satisfaction (CSAT)"].quantile(0.25),
        "high_impact_rt": data["Response Time (ms)"].quantile(
            0.75
        ),  # Higher response time is worse
        "low_impact_uptime": data["Service Uptime (%)"].quantile(0.25),
    }
    return thresholds


def plot_cost_reduction_opportunities(avg_cost_df):
    thresholds = calculate_cost_reduction_thresholds(avg_cost_df)

    # Create a scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define marker shapes and colors
    markers = {
        "EC2 Recommendation System Servers (p3.2xlarge)": "o",
        "EC2 API Servers (m5.large)": "s",
        "EC2 Streaming Servers (c5.large)": "^",
        "EC2 Database Servers (r5.large)": "D",
    }
    colors = {
        "EC2 Recommendation System Servers (p3.2xlarge)": "red",
        "EC2 API Servers (m5.large)": "orange",
        "EC2 Streaming Servers (c5.large)": "blue",
        "EC2 Database Servers (r5.large)": "green",
    }

    # Plot all data points
    for i in range(len(avg_cost_df)):
        server_config = avg_cost_df["Server Configuration"][i]
        ax.scatter(
            avg_cost_df["Operational Costs ($)"][i],
            avg_cost_df["Customer Satisfaction (CSAT)"][i],
            color=colors.get(server_config, "grey"),
            marker=markers.get(server_config, "o"),
            s=100,
            label=server_config,
        )
        ax.text(
            avg_cost_df["Operational Costs ($)"][i],
            avg_cost_df["Customer Satisfaction (CSAT)"][i],
            server_config,
            fontsize=9,
        )

    # Highlight high cost and low impact points
    high_cost_low_impact = avg_cost_df.loc[
        (avg_cost_df["Operational Costs ($)"] >= thresholds["high_cost"])
        & (avg_cost_df["Customer Satisfaction (CSAT)"] <= thresholds["low_impact_cs"])
        & (avg_cost_df["Response Time (ms)"] >= thresholds["high_impact_rt"])
        & (avg_cost_df["Service Uptime (%)"] <= thresholds["low_impact_uptime"])
    ]

    for i in range(len(high_cost_low_impact)):
        ax.scatter(
            high_cost_low_impact["Operational Costs ($)"].iloc[i],
            high_cost_low_impact["Customer Satisfaction (CSAT)"].iloc[i],
            color="red",
            marker="x",
            s=100,
        )
        ax.annotate(
            "High Cost, Low Impact",
            (
                high_cost_low_impact["Operational Costs ($)"].iloc[i],
                high_cost_low_impact["Customer Satisfaction (CSAT)"].iloc[i],
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="red",
        )

    # Annotate each server configuration with insights based on their position
    for i in range(len(avg_cost_df)):
        x, y = (
            avg_cost_df["Operational Costs ($)"][i],
            avg_cost_df["Customer Satisfaction (CSAT)"][i],
        )
        if x > thresholds["high_cost"] and y < thresholds["low_impact_cs"]:
            insight = "High Cost, Low Satisfaction"
        elif x < thresholds["high_cost"] and y > thresholds["low_impact_cs"]:
            insight = "Low Cost, High Satisfaction"
        else:
            insight = "Moderate Cost, Moderate Satisfaction"

        ax.annotate(
            insight,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color=colors.get(avg_cost_df["Server Configuration"][i], "grey"),
        )

    ax.axvline(
        thresholds["high_cost"],
        color="grey",
        linestyle="--",
        label="High Cost Threshold",
    )
    ax.axhline(
        thresholds["low_impact_cs"],
        color="grey",
        linestyle="--",
        label="Low Impact Threshold (CSAT)",
    )

    ax.set_title("Operational Cost Reduction Opportunities")
    ax.set_xlabel("Average Operational Costs ($)")
    ax.set_ylabel("Customer Satisfaction (CSAT)")
    ax.grid(True)

    # Reposition legend to the right of the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    save_plot(fig, "cost_reduction_opportunities.png")
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


# Function to plot PCA loadings
def plot_pca_loadings(loadings):
    fig, ax = plt.subplots(figsize=(14, 8))
    loadings.plot(kind="bar", ax=ax)
    ax.set_title(
        "PCA Loadings for Principal Components 1 and 2", fontsize=16, weight="bold"
    )
    ax.set_xlabel("Features", fontsize=14)
    ax.set_ylabel("Contribution to Principal Component", fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.5)
    save_plot(fig, "pca_loadings.png")
    plt.show()


def plot_business_insights_with_arrows(cluster_business_summary, business_insights):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Cluster Business Summary Insights", fontsize=16)

    # Plot Customer Satisfaction (CSAT)
    axes[0, 0].bar(
        cluster_business_summary.index,
        cluster_business_summary["Customer Satisfaction (CSAT)"],
        color="skyblue",
    )
    axes[0, 0].set_title("Customer Satisfaction (CSAT)")
    axes[0, 0].set_xlabel("Cluster")
    axes[0, 0].set_ylabel("CSAT")
    if cluster_business_summary["Customer Satisfaction (CSAT)"].min() < 90:
        min_index = cluster_business_summary["Customer Satisfaction (CSAT)"].idxmin()
        axes[0, 0].annotate(
            "Lower satisfaction here",
            xy=(
                min_index,
                cluster_business_summary["Customer Satisfaction (CSAT)"][min_index],
            ),
            xytext=(
                min_index,
                cluster_business_summary["Customer Satisfaction (CSAT)"][min_index] + 1,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Operational Costs ($)
    axes[0, 1].bar(
        cluster_business_summary.index,
        cluster_business_summary["Operational Costs ($)"],
        color="lightgreen",
    )
    axes[0, 1].set_title("Operational Costs ($)")
    axes[0, 1].set_xlabel("Cluster")
    axes[0, 1].set_ylabel("Costs ($)")
    if cluster_business_summary["Operational Costs ($)"].max() > 1100:
        max_index = cluster_business_summary["Operational Costs ($)"].idxmax()
        axes[0, 1].annotate(
            "High operational cost",
            xy=(
                max_index,
                cluster_business_summary["Operational Costs ($)"][max_index],
            ),
            xytext=(
                max_index,
                cluster_business_summary["Operational Costs ($)"][max_index] + 50,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Service Uptime (%)
    axes[1, 0].bar(
        cluster_business_summary.index,
        cluster_business_summary["Service Uptime (%)"],
        color="lightcoral",
    )
    axes[1, 0].set_title("Service Uptime (%)")
    axes[1, 0].set_xlabel("Cluster")
    axes[1, 0].set_ylabel("Uptime (%)")
    if cluster_business_summary["Service Uptime (%)"].min() < 99:
        min_index = cluster_business_summary["Service Uptime (%)"].idxmin()
        axes[1, 0].annotate(
            "Slightly lower uptime",
            xy=(min_index, cluster_business_summary["Service Uptime (%)"][min_index]),
            xytext=(
                min_index,
                cluster_business_summary["Service Uptime (%)"][min_index] + 0.3,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Response Time (ms)
    axes[1, 1].bar(
        cluster_business_summary.index,
        cluster_business_summary["Response Time (ms)"],
        color="lightblue",
    )
    axes[1, 1].set_title("Response Time (ms)")
    axes[1, 1].set_xlabel("Cluster")
    axes[1, 1].set_ylabel("Response Time (ms)")
    if cluster_business_summary["Response Time (ms)"].max() > 220:
        max_index = cluster_business_summary["Response Time (ms)"].idxmax()
        axes[1, 1].annotate(
            "High response time",
            xy=(max_index, cluster_business_summary["Response Time (ms)"][max_index]),
            xytext=(
                max_index,
                cluster_business_summary["Response Time (ms)"][max_index] + 20,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )
    elif cluster_business_summary["Response Time (ms)"].max() > 190:
        max_index = cluster_business_summary["Response Time (ms)"].idxmax()
        axes[1, 1].annotate(
            "Moderate response time",
            xy=(max_index, cluster_business_summary["Response Time (ms)"][max_index]),
            xytext=(
                max_index,
                cluster_business_summary["Response Time (ms)"][max_index] + 20,
            ),
            arrowprops=dict(facecolor="orange", shrink=0.05),
            fontsize=12,
            color="orange",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_plot(fig, "business_insights_with_arrows.png")
    plt.show()
