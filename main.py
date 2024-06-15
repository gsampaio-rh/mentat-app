import pandas as pd
import logging
from config.config import Config
from data_loader import load_data
from data_preprocessing import preprocess_data
from analysis_factory import AnalysisFactory
from visualizations.summary_statistics import plot_summary_statistics
from visualizations.metrics_plot import plot_all_metrics_single_chart
from visualizations.bubble_chart import plot_bubble_chart
from visualizations.temporal_trends import plot_temporal_trends
from visualizations.correlation_plot import (
    plot_correlation_bar,
    plot_correlation_heatmap,
)
from recommendations import Recommendations
from clustering import Clustering


def main():
    # Setup logging
    Config.setup_logging()

    # Ensure output directory exists
    Config.ensure_output_dir_exists()

    # Load data
    logging.info("Loading data...")
    operational_data, business_data = load_data(
        Config.OPERATIONAL_FILE, Config.BUSINESS_FILE
    )
    if operational_data is None or business_data is None:
        logging.error("Data loading failed. Exiting.")
        return

    # Preprocess data
    logging.info("Preprocessing data...")
    combined_data = preprocess_data(operational_data, business_data)
    if combined_data is None:
        logging.error("Data preprocessing failed. Exiting.")
        return

    # Print combined data columns for debugging
    print("Combined data columns:", combined_data.columns.tolist())

    # Perform clustering analysis
    logging.info("Performing clustering analysis...")
    clustering = Clustering()
    # Exclude non-numeric columns
    numeric_data = combined_data.drop(columns=["Timestamp", "Server Configuration"])
    clusters, kmeans_model = clustering.apply_kmeans_clustering(numeric_data)
    pca_df, pca_model = clustering.apply_pca(numeric_data, clusters)
    tsne_df = clustering.apply_tsne(numeric_data, clusters)
    best_num_clusters, best_silhouette_score, best_clusters, best_kmeans_model = (
        clustering.tune_kmeans_clustering(numeric_data)
    )
    logging.info(
        f"Best number of clusters: {best_num_clusters} with Silhouette Score: {best_silhouette_score}"
    )

    # Enrich combined data with cluster labels
    combined_data["Cluster"] = clusters
    print("Combined data columns after clustering:", combined_data.columns.tolist())

    # Perform other analyses
    logging.info("Performing other analyses...")
    analysis_factory = AnalysisFactory()

    # Ensure analysis types are correct
    summary_statistics = analysis_factory.create_analysis(
        "stat"
    ).display_summary_statistics(operational_data, operational_data.columns.tolist())
    cluster_profiles = analysis_factory.create_analysis(
        "cluster"
    ).generate_cluster_profiles(
        combined_data, operational_data.columns.tolist(), business_data.columns.tolist()
    )
    if cluster_profiles is None:
        logging.error("Cluster profile generation failed. Exiting.")
        return

    # Generate recommendations and insights
    logging.info("Generating recommendations and insights...")
    recommendations = Recommendations.generate_optimization_recommendations(
        cluster_profiles
    )
    business_insights = Recommendations.generate_business_insights(cluster_profiles)

    # Visualizations
    logging.info("Generating visualizations...")
    plot_summary_statistics(operational_data, operational_data.columns.tolist())
    plot_all_metrics_single_chart(operational_data, operational_data.columns.tolist())
    plot_bubble_chart(operational_data)
    plot_temporal_trends(operational_data, operational_data.columns.tolist())
    correlation_coefficients = analysis_factory.create_analysis(
        "correlation"
    ).calculate_correlation_coefficients(
        operational_data,
        operational_data.columns.tolist(),
        business_data.columns.tolist(),
    )
    plot_correlation_bar(
        correlation_coefficients, business_data.columns.tolist(), Config.OUTPUT_DIR
    )
    plot_correlation_heatmap(correlation_coefficients, Config.OUTPUT_DIR)

    # Log recommendations and insights
    for rec in recommendations:
        logging.info(rec)
    for insight in business_insights:
        logging.info(insight)


if __name__ == "__main__":
    main()
