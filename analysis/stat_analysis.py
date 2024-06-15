import pandas as pd
import logging


class StatAnalysis:
    def display_summary_statistics(self, data, features):
        """
        Display summary statistics for the specified features.

        Args:
        - data (pd.DataFrame): DataFrame containing the data.
        - features (list): List of features to display summary statistics for.
        """
        try:
            summary = data[features].describe()
            print("Summary Statistics:")
            print(summary)
            logging.info("Summary statistics displayed successfully.")
        except Exception as e:
            logging.error(f"An error occurred while displaying summary statistics: {e}")

    def analyze_distributions(self, data, features, cluster_col="cluster"):
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
        try:
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
            logging.info("Distributions and variances analyzed successfully.")
        except Exception as e:
            logging.error(f"An error occurred while analyzing distributions: {e}")
        return analysis_results
