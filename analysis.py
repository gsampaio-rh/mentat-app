# analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot
import os


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


def generate_cluster_profiles(enriched_data, features, business_metrics):
    """
    Generate profiles for each cluster with summary statistics.

    Args:
    - enriched_data (pd.DataFrame): DataFrame containing the enriched data with cluster assignments.
    - features (list): List of operational features.
    - business_metrics (list): List of business metrics.

    Returns:
    - pd.DataFrame: DataFrame containing the cluster profiles.
    """
    cluster_profiles = []

    # Define default values for all expected keys
    default_profile = {
        "Cluster": None,
        "Mean CPU Utilization (%)": 0,
        "Mean Memory Utilization (%)": 0,
        "Mean Network I/O Throughput (Mbps)": 0,
        "Mean Disk I/O Throughput (MB/s)": 0,
        "Mean Operational Costs ($)": 0,
        "Mean Customer Satisfaction (CSAT)": 0,
        "Mean Service Uptime (%)": 0,
        "Mean Response Time (ms)": 0,
    }

    # Compute profiles for each cluster
    for cluster_id in enriched_data["Cluster"].unique():
        cluster_data = enriched_data[enriched_data["Cluster"] == cluster_id]

        # Start with default profile
        profile = default_profile.copy()
        profile["Cluster"] = cluster_id

        # Update with actual values
        profile.update(cluster_data[features + business_metrics].mean().to_dict())

        cluster_profiles.append(profile)

    cluster_profiles_df = pd.DataFrame(cluster_profiles)
    return cluster_profiles_df


def generate_optimization_recommendations(cluster_profiles):
    """
    Generate optimization recommendations based on cluster profiles.

    Args:
    - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles with summary statistics.

    Returns:
    - list: List of recommendations.
    """
    recommendations = []

    for idx, profile in cluster_profiles.iterrows():
        cluster_id = profile["Cluster"]
        recommendation = f"Cluster {cluster_id} Recommendations:\n"

        if profile.get("Mean CPU Utilization (%)", 0) > 80:
            recommendation += "- CPU utilization is high. Consider load balancing or upgrading CPU capacity.\n"

        if profile.get("Mean Memory Utilization (%)", 0) > 80:
            recommendation += "- Memory utilization is high. Consider optimizing memory usage or upgrading memory capacity.\n"

        if profile.get("Mean Network I/O Throughput (Mbps)", 0) > 1000:
            recommendation += "- Network I/O throughput is high. Ensure network bandwidth is sufficient.\n"

        if profile.get("Mean Disk I/O Throughput (MB/s)", 0) > 500:
            recommendation += "- Disk I/O throughput is high. Consider using faster storage solutions.\n"

        if profile.get("Mean Operational Costs ($)", 0) > 10000:
            recommendation += (
                "- Operational costs are high. Look into cost-saving measures.\n"
            )

        if profile.get("Mean Customer Satisfaction (CSAT)", 0) < 70:
            recommendation += "- Customer satisfaction is low. Investigate and address customer issues.\n"

        if profile.get("Mean Service Uptime (%)", 0) < 99:
            recommendation += (
                "- Service uptime is below the target. Improve system reliability.\n"
            )

        if profile.get("Mean Response Time (ms)", 0) > 200:
            recommendation += (
                "- Response time is high. Optimize performance to reduce latency.\n"
            )

        recommendations.append(recommendation.strip())

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

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(12, 8))  # Keep the figure size fixed

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    ax.set_title("Correlation Matrix of Business Metrics and Resources", fontsize=16)

    # Add custom color bar label
    cbar = ax.collections[0].colorbar
    cbar.set_label(
        "Correlation Coefficient\n(Higher positive correlation to Higher inverse correlation)",
        rotation=270,
        labelpad=30,
    )

    # Adjust the rotation and alignment of the x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust the rotation and alignment of the y-axis labels if necessary
    plt.yticks(rotation=0)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()

    # Save the plot
    save_plot(fig, "correlation_matrix.png")
    plt.show()

    return correlation_matrix


def calculate_correlation_coefficients(data, features, business_metrics, output_dir):
    """
    Calculate correlation coefficients for the provided data and save the results.

    :param data: DataFrame containing the data
    :param features: List of feature columns
    :param business_metrics: List of business metric columns
    :param output_dir: Directory to save the output
    :return: DataFrame of correlation coefficients
    """
    correlation_coefficients = data[features + business_metrics].corr()
    print("\nCorrelation Coefficients:\n", correlation_coefficients)
    correlation_coefficients.to_csv(
        os.path.join(output_dir, "correlation_coefficients.csv")
    )
    return correlation_coefficients


def identify_key_drivers(
    correlation_coefficients, business_metrics, output_dir, threshold=0.5
):
    """
    Identify key drivers for each business metric and save the results.

    :param correlation_coefficients: DataFrame of correlation coefficients
    :param business_metrics: List of business metric columns
    :param output_dir: Directory to save the output
    :param threshold: Correlation coefficient threshold to consider a variable as a key driver
    """
    for metric in business_metrics:
        print(f"\nKey Drivers for {metric}:")
        sorted_correlations = (
            correlation_coefficients[metric].abs().sort_values(ascending=False)
        )
        key_drivers = sorted_correlations[
            sorted_correlations > threshold
        ].index.tolist()
        key_drivers.remove(metric)  # Remove the metric itself from the list
        print(key_drivers)
        with open(os.path.join(output_dir, f"key_drivers_{metric}.txt"), "w") as f:
            for driver in key_drivers:
                f.write(f"{driver}\n")
