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
    scaled_operational_data, scaled_business_data = preprocess_data(
        operational_data, business_data
    )
    if scaled_operational_data is None or scaled_business_data is None:
        logging.error("Data preprocessing failed. Exiting.")
        return

    # Perform clustering analysis
    logging.info("Performing clustering analysis...")
    clustering = Clustering()
    clusters, kmeans_model = clustering.apply_kmeans_clustering(scaled_operational_data)
    pca_df, pca_model = clustering.apply_pca(scaled_operational_data, clusters)
    tsne_df = clustering.apply_tsne(scaled_operational_data, clusters)
    best_num_clusters, best_silhouette_score, best_clusters, best_kmeans_model = (
        clustering.tune_kmeans_clustering(scaled_operational_data)
    )

    # Perform other analyses
    logging.info("Performing other analyses...")
    analysis_factory = AnalysisFactory()
    summary_statistics = analysis_factory.create_analysis(
        "stat"
    ).display_summary_statistics(operational_data, operational_data.columns.tolist())
    cluster_profiles = analysis_factory.create_analysis(
        "cluster"
    ).generate_cluster_profiles(
        pca_df, operational_data.columns.tolist(), business_data.columns.tolist()
    )

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
