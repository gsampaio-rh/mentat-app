# main.py

from data_loader import read_csv_file
from data_preprocessing import (
    normalize_data,
    clean_and_scale_data,
    preprocess_for_utilization,
    preprocess_for_cost_reduction,
    add_new_features,
)
from visualization import (
    plot_summary_statistics,
    plot_correlation_heatmap, 
    plot_correlation_bar, 
    plot_pair_plots, 
    plot_key_driver_impact
)

from clustering import (
    apply_kmeans_clustering,
    apply_pca,
    get_pca_loadings,
    apply_tsne,
    tune_kmeans_clustering,
)

from analysis import (
    display_summary_statistics,
    analyze_distributions,
    generate_optimization_recommendations,
    calculate_cluster_business_summary,
    generate_business_insights,
    generate_cluster_profiles,
    identify_best_worst_clusters,
    generate_correlation_matrix,
    calculate_correlation_coefficients,  # Add this import
    identify_key_drivers,  # Add this import
)

# Import silhouette_score
from sklearn.metrics import silhouette_score

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

# Additional Metrics
ADDITIONAL_METRICS = [
    'CPU_to_Network_Ratio', 
    'Memory_to_Disk_Ratio', 
    'Cost_per_Throughput', 
    'Uptime_Efficiency', 
    'Satisfaction_per_Dollar', 
    'Cost_per_Uptime', 
    'Response_per_Dollar', 
    'Cost_Efficiency_Index',
    'Hourly_Avg_Satisfaction',
    'Hourly_Total_Costs'
]


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

    # Print initial dataset summaries
    print("\nInitial Operational Data Summary")
    display_summary_statistics(operational_data, FEATURES)
    print("\nInitial Business Data Summary")
    display_summary_statistics(business_data, BUSINESS_METRICS)

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

    # Step 5: Feature Engineering
    enriched_data = add_new_features(cleaned_data)
    print("Enriched Data Columns:", enriched_data.columns.tolist())
    print("Enriched Data Head:\n", enriched_data.head())

    # Step 6: Display Final KPIs
    print("Data Preprocessing Completed Successfully.")

    # Plot summary statistics from cleaned and enriched data
    plot_summary_statistics(
        enriched_data,
        FEATURES + BUSINESS_METRICS + ADDITIONAL_METRICS,
    )

    # Step 7: Generate and plot correlation matrix
    correlation_matrix = generate_correlation_matrix(
        enriched_data,
        FEATURES + BUSINESS_METRICS + ADDITIONAL_METRICS,
    )
    print("\nCorrelation Matrix:\n", correlation_matrix)
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))

    # Step 8: Plot pair plots
    # plot_pair_plots(
    #     enriched_data,
    #     FEATURES + BUSINESS_METRICS + ADDITIONAL_METRICS,
    # )

    # Step 9: Tune the number of clusters
    best_num_clusters, best_silhouette_score, best_clusters, best_kmeans_model = (
        tune_kmeans_clustering(scaled_data[: len(cleaned_data)])
    )
    enriched_data["Cluster"] = best_clusters

    # Step 10: Use the best number of clusters
    enriched_data["Cluster"] = best_clusters

    # Step 11: Apply PCA and plot
    pca_df, pca_model = apply_pca(scaled_data[: len(cleaned_data)], best_clusters)

    # Step 12: Get and save PCA loadings
    pca_loadings = get_pca_loadings(pca_model, CLUSTERING_FEATURES)
    pca_loadings.to_csv(os.path.join(OUTPUT_DIR, "pca_loadings.csv"))

    # Step 13: Apply t-SNE and plot
    tsne_df = apply_tsne(scaled_data[: len(cleaned_data)], best_clusters)

    # Step 14: Correlation Analysis and Key Driver Identification
    print("\nStep 14: Correlation Analysis and Key Driver Identification")

    # Calculate correlation coefficients
    correlation_coefficients = calculate_correlation_coefficients(
        enriched_data, FEATURES, BUSINESS_METRICS, OUTPUT_DIR
    )

    # Identify key drivers of business metrics
    identify_key_drivers(correlation_coefficients, BUSINESS_METRICS, OUTPUT_DIR)

    # Visualize correlation heatmap
    plot_correlation_heatmap(correlation_coefficients, OUTPUT_DIR)

    # Visualize correlation bar plots for key drivers
    plot_correlation_bar(correlation_coefficients, BUSINESS_METRICS, OUTPUT_DIR)

    # Visualize pair plots for each business metric and its key drivers
    for metric in BUSINESS_METRICS:
        key_drivers = (
            correlation_coefficients[metric].abs().sort_values(ascending=False)
        )
        key_drivers = key_drivers[key_drivers > 0.5].index.tolist()
        key_drivers.remove(metric)
        plot_pair_plots(enriched_data, key_drivers + [metric], OUTPUT_DIR, metric)

    # Visualize key driver impact over time
    for metric in BUSINESS_METRICS:
        key_drivers = (
            correlation_coefficients[metric].abs().sort_values(ascending=False)
        )
        key_drivers = key_drivers[key_drivers > 0.5].index.tolist()
        key_drivers.remove(metric)
        plot_key_driver_impact(enriched_data, key_drivers, metric, OUTPUT_DIR)

    # Step 15: Generate Optimization Recommendations
    print("\nStep 15: Generate Optimization Recommendations")
    cluster_profiles = generate_cluster_profiles(
        enriched_data, FEATURES, BUSINESS_METRICS
    )
    recommendations = generate_optimization_recommendations(cluster_profiles)
    with open(os.path.join(OUTPUT_DIR, "optimization_recommendations.txt"), "w") as f:
        for recommendation in recommendations:
            f.write(recommendation + "\n")
    print("Optimization Recommendations:\n", "\n".join(recommendations))

    # Step 16: Generate Business Insights
    print("\nStep 16: Generate Business Insights")
    insights = generate_business_insights(cluster_profiles)
    with open(os.path.join(OUTPUT_DIR, "business_insights.txt"), "w") as f:
        for insight in insights:
            f.write(insight + "\n")
    print("Business Insights:\n", "\n".join(insights))


if __name__ == "__main__":
    main()
