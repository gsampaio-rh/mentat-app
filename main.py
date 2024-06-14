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

# Combined features for clustering
CLUSTERING_FEATURES = FEATURES + BUSINESS_METRICS

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
    normalized_operational_data = normalize_data(operational_data, FEATURES)
    normalized_business_data = normalize_data(business_data, BUSINESS_METRICS)

    print("Operational Data Columns:", normalized_operational_data.columns.tolist())
    print("Business Data Columns:", normalized_business_data.columns.tolist())

    # Merge the data on Timestamp and Server Configuration
    combined_data = pd.merge(normalized_operational_data, normalized_business_data, on=["Timestamp", "Server Configuration"])

    # Display the head of the merged data to verify
    print("Combined Data Columns:", combined_data.columns.tolist())
    print("Combined Data Head:\n", combined_data.head())

    # Clean and scale the data
    cleaned_data, scaled_data = clean_and_scale_data(combined_data, CLUSTERING_FEATURES)

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

    # Visualize clusters
    # visualize_clusters(cleaned_data, "cluster", FEATURES + BUSINESS_METRICS)

    # Apply PCA and plot the results
    pca_df, pca = apply_pca(scaled_data, clustered_data)
    pca_loadings = get_pca_loadings(pca, FEATURES)
    print("PCA Loadings:\n", pca_loadings)


if __name__ == "__main__":
    main()
