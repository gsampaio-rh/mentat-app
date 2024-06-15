# main.py

from data_loader import read_csv_file
from data_preprocessing import (
    normalize_data,
    clean_and_scale_data,
    normalize_profiles,
    preprocess_for_utilization,
    preprocess_for_cost_reduction,
)
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
    plot_cost_reduction_opportunities,
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

# Combined features for clustering
CLUSTERING_FEATURES = FEATURES + BUSINESS_METRICS


def main():
    # File paths
    operational_file_path = "data/netflix_operational_metrics.csv"
    business_file_path = "data/netflix_business_metrics.csv"

    # Step 1: Read Data
    operational_data = read_csv_file(operational_file_path)
    business_data = read_csv_file(business_file_path)
    print("Successfully read the files.")
    print("Operational Data Columns:", operational_data.columns.tolist())
    print("Business Data Columns:", business_data.columns.tolist())

    # Step 2: Normalize Data
    normalized_operational_data = normalize_data(operational_data, FEATURES)
    normalized_business_data = normalize_data(business_data, BUSINESS_METRICS)
    print(
        "Normalized Operational Data Columns:",
        normalized_operational_data.columns.tolist(),
    )
    print(
        "Normalized Business Data Columns:", normalized_business_data.columns.tolist()
    )

    # Step 3: Merge Data
    combined_data = pd.merge(
        normalized_operational_data,
        normalized_business_data,
        on=["Timestamp", "Server Configuration"],
    )
    print("Combined Data Columns:", combined_data.columns.tolist())
    print("Combined Data Head:\n", combined_data.head())

    # Step 4: Clean and Scale Data
    cleaned_data, scaled_data = clean_and_scale_data(combined_data, CLUSTERING_FEATURES)
    print("Cleaned Data Shape:", cleaned_data.shape)

    # Step 5: Display Final KPIs
    print("Data Preprocessing Completed Successfully.")

    # Apply K-Means clustering
    clustered_data, kmeans_model = apply_kmeans_clustering(scaled_data, num_clusters=5)

    # Check the clustered data
    print("Clustered Data Head:\n", clustered_data[:5])  # Use slicing for numpy array

    # Add the cluster labels to the cleaned data
    cleaned_data["cluster"] = clustered_data

    # Verify the 'Cluster' column addition
    print("Cleaned Data Columns after adding 'Cluster':", cleaned_data.columns.tolist())

    # Generate cluster profiles
    cluster_profiles = generate_cluster_profiles(
        cleaned_data, FEATURES, BUSINESS_METRICS
    )

    print("Cluster Profiles:\n", cluster_profiles)

    # Normalize cluster profiles for plotting
    normalized_cluster_profiles = normalize_data(
        cluster_profiles, FEATURES + BUSINESS_METRICS
    )

    # Save cluster profiles
    cluster_profiles.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))

    # Plot summary statistics
    plot_summary_statistics(combined_data, FEATURES + BUSINESS_METRICS)

    # Plot temporal trends
    plot_temporal_trends(business_data, BUSINESS_METRICS)

    # Visualize resource utilization efficiency
    avg_utilization_df = preprocess_for_utilization(combined_data)
    plot_resource_utilization_efficiency(avg_utilization_df)

    # Preprocess for cost reduction analysis
    avg_cost_df = preprocess_for_cost_reduction(combined_data)

    # Plot cost reduction opportunities
    plot_cost_reduction_opportunities(avg_cost_df)

    # Apply PCA and plot the results
    pca_df, pca = apply_pca(scaled_data, clustered_data)
    pca_loadings = get_pca_loadings(pca, CLUSTERING_FEATURES)
    print("PCA Loadings:\n", pca_loadings)

    # Save PCA loadings
    pca_loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))


if __name__ == "__main__":
    main()
