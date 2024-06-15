import pandas as pd
import logging


class ClusteringAnalysis:
    """
    Class for performing clustering analysis.
    """

    def generate_cluster_profiles(self, enriched_data, features, business_metrics):
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

        try:
            # Verify column names
            expected_columns = features + business_metrics
            missing_columns = [col for col in expected_columns if col not in enriched_data.columns]
            if missing_columns:
                logging.error(f"Missing columns in enriched_data: {missing_columns}")
                return pd.DataFrame()

            for cluster_id in enriched_data["Cluster"].unique():
                cluster_data = enriched_data[enriched_data["Cluster"] == cluster_id]
                profile = default_profile.copy()
                profile["Cluster"] = cluster_id
                profile.update(
                    cluster_data[features + business_metrics].mean().to_dict()
                )
                cluster_profiles.append(profile)
            logging.info("Cluster profiles generated successfully.")
            return pd.DataFrame(cluster_profiles)
        except Exception as e:
            logging.error(f"An error occurred while generating cluster profiles: {e}")
            return pd.DataFrame()

    def calculate_cluster_business_summary(
        self, data, business_metrics, cluster_col="cluster"
    ):
        """
        Calculate the mean of business metrics for each cluster.

        Args:
        - data (pd.DataFrame): DataFrame containing the business metrics and cluster labels.
        - business_metrics (list): List of business metrics to calculate.
        - cluster_col (str): Name of the cluster column.

        Returns:
        - pd.DataFrame: DataFrame containing the mean of business metrics for each cluster.
        """
        try:
            cluster_business_summary = data.groupby(cluster_col)[
                business_metrics
            ].mean()
            logging.info("Cluster business summary calculated successfully.")
            return cluster_business_summary
        except Exception as e:
            logging.error(
                f"An error occurred while calculating cluster business summary: {e}"
            )
            return pd.DataFrame()

    def identify_best_worst_clusters(self, cluster_profiles, metric):
        """
        Identify the best and worst performing clusters based on the specified metric.

        Args:
        - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.
        - metric (str): Metric to use for identifying best and worst clusters.

        Returns:
        - tuple: Best performing cluster, worst performing cluster.
        """
        try:
            best_cluster = cluster_profiles[metric].idxmax()
            worst_cluster = cluster_profiles[metric].idxmin()
            logging.info("Best and worst clusters identified successfully.")
            return best_cluster, worst_cluster
        except Exception as e:
            logging.error(f"An error occurred while identifying clusters: {e}")
            return None, None
