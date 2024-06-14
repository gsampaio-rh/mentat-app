# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot


def display_summary_statistics(data, features):
    """
    Display summary statistics for the specified features.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to display summary statistics for.
    """
    summary = data[features].describe()
    print("Summary Statistics:")
    print(summary)


def analyze_distributions(data, features, cluster_col="cluster"):
    """
    Analyze distributions and variances of features within clusters.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to analyze.
    - cluster_col (str): Name of the cluster column.

    Returns:
    - dict: Dictionary containing analysis results.
    """
    analysis_results = {}
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        cluster_analysis = {}
        for feature in features:
            distribution = cluster_data[feature].describe()
            variance = cluster_data[feature].var()
            cluster_analysis[feature] = {
                "distribution": distribution,
                "variance": variance,
            }
        analysis_results[cluster] = cluster_analysis
    return analysis_results


def generate_optimization_recommendations(cluster_profiles):
    """
    Generate optimization recommendations based on cluster profiles.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.

    Returns:
    - list: List of optimization recommendations.
    """
    recommendations = []
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        recommendations.append(f"Cluster {cluster} Recommendations:")

        if profile["CPU Utilization (%)"] > 80:
            recommendations.append(
                " - High CPU utilization. Consider load balancing or upgrading CPU resources."
            )
        elif profile["CPU Utilization (%)"] < 20:
            recommendations.append(
                " - Low CPU utilization. Evaluate if resources can be scaled down to save costs."
            )

        if profile["Memory Utilization (%)"] > 80:
            recommendations.append(
                " - High memory utilization. Consider adding more memory or optimizing memory usage."
            )
        elif profile["Memory Utilization (%)"] < 20:
            recommendations.append(
                " - Low memory utilization. Evaluate if resources can be scaled down to save costs."
            )

        if profile["Network I/O Throughput (Mbps)"] > 1000:
            recommendations.append(
                " - High network throughput. Ensure network infrastructure can handle the load."
            )
        elif profile["Network I/O Throughput (Mbps)"] < 100:
            recommendations.append(
                " - Low network throughput. Evaluate if network resources are over-provisioned."
            )

        if profile["Disk I/O Throughput (MB/s)"] > 500:
            recommendations.append(
                " - High disk I/O. Consider using faster storage solutions or optimizing disk access patterns."
            )
        elif profile["Disk I/O Throughput (MB/s)"] < 50:
            recommendations.append(
                " - Low disk I/O. Evaluate if storage resources are over-provisioned."
            )

        if profile["Service Uptime (%)"] < 95:
            recommendations.append(
                " - Low service uptime. Investigate causes of downtime and improve reliability."
            )

        if profile["Response Time (ms)"] > 1000:
            recommendations.append(
                " - High response time. Optimize application performance to reduce response time."
            )

        recommendations.append("")  # Add a newline for better readability

    return recommendations


def calculate_cluster_business_summary(data, business_metrics, cluster_col="cluster"):
    """
    Calculate the mean of business metrics for each cluster.

    Args:
    - data (pd.DataFrame): DataFrame containing the business metrics and cluster labels.
    - business_metrics (list): List of business metrics to calculate.
    - cluster_col (str): Name of the cluster column.

    Returns:
    - pd.DataFrame: DataFrame containing the mean of business metrics for each cluster.
    """
    cluster_business_summary = data.groupby(cluster_col)[business_metrics].mean()
    return cluster_business_summary


def generate_business_insights(cluster_profiles):
    """
    Generate actionable business insights based on cluster profiles.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.

    Returns:
    - list: List of business insights.
    """
    insights = []
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        insights.append(f"Cluster {cluster} Insights:")
        if profile["Customer Satisfaction (CSAT)"] > 80:
            insights.append(
                " - High customer satisfaction. Continue current practices."
            )
        elif profile["Customer Satisfaction (CSAT)"] < 50:
            insights.append(
                " - Low customer satisfaction. Investigate and address issues."
            )

        if profile["Operational Costs ($)"] > 10000:
            insights.append(
                " - High operational costs. Look for optimization opportunities."
            )
        elif profile["Operational Costs ($)"] < 5000:
            insights.append(
                " - Low operational costs. Evaluate if cost savings are affecting performance."
            )

        if profile["Service Uptime (%)"] < 95:
            insights.append(
                " - Low service uptime. Improve reliability and reduce downtimes."
            )

        insights.append("")  # Add a newline for better readability

    return insights


def generate_cluster_profiles(data, features, business_metrics, cluster_col="cluster"):
    """
    Generate profiles for each cluster based on the specified features and business metrics.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to include in the profiles.
    - business_metrics (list): List of business metrics to include in the profiles.
    - cluster_col (str): Name of the cluster column.

    Returns:
    - pd.DataFrame: DataFrame containing the cluster profiles.
    """
    cluster_profiles = data.groupby(cluster_col)[features + business_metrics].mean()
    print("Cluster Profiles:")
    print(cluster_profiles)
    return cluster_profiles


def identify_best_worst_clusters(cluster_profiles, metric):
    """
    Identify the best and worst performing clusters based on the specified metric.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.
    - metric (str): Metric to use for identifying best and worst clusters.

    Returns:
    - tuple: Best performing cluster, worst performing cluster.
    """
    best_cluster = cluster_profiles[metric].idxmax()
    worst_cluster = cluster_profiles[metric].idxmin()
    return best_cluster, worst_cluster


def generate_correlation_matrix(data, features):
    """
    Generate and plot a correlation matrix for the specified features.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to include in the correlation matrix.

    Returns:
    - pd.DataFrame: DataFrame containing the correlation matrix.
    """
    # Check if the specified features are in the DataFrame
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not found in data columns")

    # Calculate the correlation matrix
    correlation_matrix = data[features].corr()

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Business Metrics and Resources", fontsize=16)

    # Save the plot
    save_plot(fig, "correlation_matrix.png")
    plt.close(fig)  # Close the figure to avoid displaying it inline

    return correlation_matrix
