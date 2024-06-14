# main.py

from data_loader import read_csv_file
from data_preprocessing import normalize_data, clean_and_scale_data, normalize_profiles
from visualization import (
    plot_summary_statistics,
    plot_all_metrics_single_chart,
    plot_bubble_chart,
    visualize_clusters,
    plot_temporal_trends,
    plot_resource_utilization_efficiency,
    plot_cost_benefit_analysis,
    plot_server_config_metrics,
    plot_cost_vs_performance,
    plot_pca_loadings,
    plot_business_insights_with_arrows,
)
from clustering import apply_kmeans_clustering, apply_pca, get_pca_loadings, apply_tsne
from analysis import (
    display_summary_statistics,
    analyze_distributions,
    generate_optimization_recommendations,
    calculate_cluster_business_summary,
    generate_business_insights,
    generate_cluster_profiles,
    identify_best_worst_clusters,
    generate_correlation_matrix,
)

import pandas as pd
import os

# Ensure the output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define constants for features and business metrics
FEATURES = [
    "CPU Utilization (%)",
    "Memory Utilization (%)",
    "Network I/O Throughput (Mbps)",
    "Disk I/O Throughput (MB/s)",
]
BUSINESS_METRICS = [
    "Customer Satisfaction (CSAT)",
    "Operational Costs ($)",
    "Service Uptime (%)",
    "Response Time (ms)",
]


def main():
    # Read data from CSV files
    operational_file_path = "data/netflix_operational_metrics.csv"  # Update this path to your actual CSV file
    business_file_path = (
        "data/netflix_business_metrics.csv"  # Update this path to your actual CSV file
    )

    operational_data = read_csv_file(operational_file_path)
    business_data = read_csv_file(business_file_path)

    # Ensure the columns are correctly read
    print("Operational Data Columns:", operational_data.columns.tolist())
    print("Business Data Columns:", business_data.columns.tolist())

    # Step 1: Normalize the data
    normalized_data = normalize_data(operational_data, FEATURES)

    # Plot the combined bubble chart
    plot_all_metrics_single_chart(normalized_data, FEATURES)

    # Plot the bubble chart
    plot_bubble_chart(operational_data)

    # Step 2: Clean and Scale Data
    operational_data, X_scaled_cleaned = clean_and_scale_data(
        operational_data, FEATURES
    )

    # Step 3: Apply K-Means Clustering
    clusters_cleaned, kmeans = apply_kmeans_clustering(X_scaled_cleaned)
    operational_data["cluster"] = clusters_cleaned

    # Step 4: Visualize Clusters
    visualize_clusters(operational_data, FEATURES, clusters_cleaned)

    # Step 5: Apply PCA and plot the results
    pca_df, pca = apply_pca(X_scaled_cleaned, clusters_cleaned)
    pca_loadings = get_pca_loadings(pca, FEATURES)
    print("PCA Loadings:")
    print(pca_loadings)

    # Save PCA loadings
    pca_loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))

    # Plot PCA Loadings
    plot_pca_loadings(pca_loadings)

    # Step 6: Analyze Distributions and Variances within Clusters
    analysis_results = analyze_distributions(operational_data, FEATURES)

    # Step 7: Generate Optimization Recommendations
    optimization_recommendations = generate_optimization_recommendations(
        analysis_results
    )
    with open(os.path.join(OUTPUT_DIR, "optimization_recommendations.txt"), "w") as f:
        for rec in optimization_recommendations:
            f.write(rec + "\n")

    # Ensure the cluster labels are also in the business data
    business_data["cluster"] = operational_data["cluster"]

    # Step 8: Calculate Cluster Business Summary
    cluster_business_summary = calculate_cluster_business_summary(
        business_data, BUSINESS_METRICS
    )
    print("Cluster Business Summary:")
    print(cluster_business_summary)

    # Save cluster business summary
    cluster_business_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_business_summary.csv")
    )

    # Plot Business Insights with Arrows
    plot_business_insights_with_arrows(
        cluster_business_summary, generate_business_insights(cluster_business_summary)
    )

    # Merge operational and business data for correlation analysis
    combined_data = pd.merge(
        operational_data,
        business_data,
        on=["Timestamp", "Server Configuration", "cluster"],
    )

    # Step 9: Generate Correlation Matrix
    correlation_matrix = generate_correlation_matrix(
        combined_data, BUSINESS_METRICS + FEATURES
    )

    # Save correlation matrix
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

    # Step 10: Plot Temporal Trends of Business Metrics
    # plot_temporal_trends(business_data, BUSINESS_METRICS)

    # Step 11: Generate Cluster Profiles
    cluster_profiles = generate_cluster_profiles(
        combined_data, FEATURES, BUSINESS_METRICS
    )

    # Normalize cluster profiles for plotting
    normalized_cluster_profiles = normalize_profiles(cluster_profiles)

    # Step 12: Identify Best and Worst Performing Clusters
    best_cluster, worst_cluster = identify_best_worst_clusters(
        cluster_profiles, "Customer Satisfaction (CSAT)"
    )
    print(f"Best Performing Cluster: {best_cluster}")
    print(f"Worst Performing Cluster: {worst_cluster}")

    # Save cluster profiles
    cluster_profiles.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))

    # Step 13: Generate Business Insights
    business_insights = generate_business_insights(cluster_business_summary)
    with open(os.path.join(OUTPUT_DIR, "business_insights.txt"), "w") as f:
        for insight in business_insights:
            f.write(insight + "\n")

    # Step 14: Plot Resource Utilization Efficiency by Cluster
    plot_resource_utilization_efficiency(normalized_cluster_profiles)

    # Step 15: Plot Cost-Benefit Analysis by Cluster
    plot_cost_benefit_analysis(cluster_profiles)

    # Aggregate metrics by Server Configuration
    server_config_summary = combined_data.groupby("Server Configuration")[
        FEATURES + BUSINESS_METRICS
    ].mean()

    # Normalize server configuration summary for plotting
    normalized_server_config_summary = normalize_profiles(server_config_summary)

    # Step 16: Plot Performance Metrics by Server Configuration
    plot_server_config_metrics(normalized_server_config_summary)

    # Step 17: Plot Cost vs Performance Metrics by Server Configuration
    plot_cost_vs_performance(server_config_summary)


if __name__ == "__main__":
    main()
