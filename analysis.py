# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def generate_optimization_recommendations(analysis_results):
    """
    Generate optimization recommendations based on analysis results.

    Args:
    - analysis_results (dict): Dictionary containing analysis results.

    Returns:
    - list: List of optimization recommendations.
    """
    recommendations = []
    for cluster, metrics in analysis_results.items():
        rec = f"Cluster {cluster} Optimization Recommendations:\n"
        for feature, stats in metrics.items():
            if stats["variance"] > 50:
                rec += f"- {feature}: High variance detected, consider optimizing this metric.\n"
            if feature == "CPU Utilization (%)" and stats["distribution"]["mean"] > 80:
                rec += f"- {feature}: High average utilization, consider adding more CPU resources.\n"
            if (
                feature == "Memory Utilization (%)"
                and stats["distribution"]["mean"] > 70
            ):
                rec += f"- {feature}: High average utilization, consider adding more memory or optimizing memory usage.\n"
            if (
                feature == "Network I/O Throughput (Mbps)"
                and stats["distribution"]["mean"] > 30000
            ):
                rec += f"- {feature}: High network throughput, consider increasing network bandwidth.\n"
            if (
                feature == "Disk I/O Throughput (MB/s)"
                and stats["distribution"]["mean"] > 100
            ):
                rec += f"- {feature}: High disk throughput, consider using faster storage solutions.\n"
        recommendations.append(rec)
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


def generate_business_insights(cluster_business_summary):
    """
    Generate business insights based on the cluster business summary.

    Args:
    - cluster_business_summary (pd.DataFrame): DataFrame containing the cluster business summary.

    Returns:
    - list: List of business insights.
    """
    insights = []
    for cluster in cluster_business_summary.index:
        cluster_insights = f"\nCluster {cluster} Insights:"
        if cluster_business_summary.loc[cluster, "Response Time (ms)"] > 220:
            cluster_insights += "\n- High response time detected, consider optimizing server configurations to reduce response time."
        elif cluster_business_summary.loc[cluster, "Response Time (ms)"] > 190:
            cluster_insights += "\n- Response time could be improved. Analyze server performance and network latency."
        if cluster_business_summary.loc[cluster, "Customer Satisfaction (CSAT)"] < 90:
            cluster_insights += "\n- Customer satisfaction is lower than other clusters, investigate potential causes and gather customer feedback."
        if cluster_business_summary.loc[cluster, "Operational Costs ($)"] > 1100:
            cluster_insights += "\n- High operational costs detected, consider optimizing resource usage to reduce costs."
        elif cluster_business_summary.loc[cluster, "Operational Costs ($)"] > 1000:
            cluster_insights += "\n- Operational costs are relatively high. Investigate cost-saving measures without compromising service quality."
        if cluster_business_summary.loc[cluster, "Service Uptime (%)"] < 99:
            cluster_insights += "\n- Service uptime could be improved. Investigate and address potential causes to enhance reliability."
        insights.append(cluster_insights)
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
    correlation_matrix = data[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Business Metrics and Resources", fontsize=16)
    plt.show()
    return correlation_matrix
