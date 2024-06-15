import pandas as pd
import logging


class CorrelationAnalysis:
    """
    Class for performing correlation analysis.
    """

    def generate_correlation_matrix(self, data, features):
        """
        Generate and plot a correlation matrix for the specified features.

        Args:
        - data (pd.DataFrame): DataFrame containing the data.
        - features (list): List of features to include in the correlation matrix.

        Returns:
        - pd.DataFrame: DataFrame containing the correlation matrix.
        """
        try:
            correlation_matrix = data[features].corr()
            logging.info("Correlation matrix generated successfully.")
            return correlation_matrix
        except Exception as e:
            logging.error(
                f"An error occurred while generating the correlation matrix: {e}"
            )
            return pd.DataFrame()

    def calculate_correlation_coefficients(self, data, features, business_metrics):
        """
        Calculate correlation coefficients for the provided data.

        Args:
        - data (pd.DataFrame): DataFrame containing the data.
        - features (list): List of feature columns.
        - business_metrics (list): List of business metric columns.

        Returns:
        - pd.DataFrame: DataFrame of correlation coefficients.
        """
        try:
            correlation_coefficients = data[features + business_metrics].corr()
            logging.info("Correlation coefficients calculated successfully.")
            return correlation_coefficients
        except Exception as e:
            logging.error(
                f"An error occurred while calculating correlation coefficients: {e}"
            )
            return pd.DataFrame()

    def identify_key_drivers(
        self, correlation_coefficients, business_metrics, threshold=0.5
    ):
        """
        Identify key drivers for each business metric.

        Args:
        - correlation_coefficients (pd.DataFrame): DataFrame of correlation coefficients.
        - business_metrics (list): List of business metric columns.
        - threshold (float): Correlation coefficient threshold to consider a variable as a key driver.

        Returns:
        - dict: Dictionary of key drivers for each business metric.
        """
        key_drivers = {}
        try:
            for metric in business_metrics:
                sorted_correlations = (
                    correlation_coefficients[metric].abs().sort_values(ascending=False)
                )
                key_drivers[metric] = sorted_correlations[
                    sorted_correlations > threshold
                ].index.tolist()
                if metric in key_drivers[metric]:
                    key_drivers[metric].remove(metric)
            logging.info("Key drivers identified successfully.")
            return key_drivers
        except Exception as e:
            logging.error(f"An error occurred while identifying key drivers: {e}")
            return key_drivers
