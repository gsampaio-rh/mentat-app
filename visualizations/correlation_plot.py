import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import save_plot
import logging
import os

def plot_correlation_bar(correlation_coefficients, business_metrics, output_dir):
    """
    Plot bar charts for correlation coefficients of the specified business metrics.

    Args:
    - correlation_coefficients (pd.DataFrame): DataFrame containing the correlation coefficients.
    - business_metrics (list): List of business metrics to plot.
    - output_dir (str): Directory to save the plots.
    """
    try:
        for metric in business_metrics:
            sorted_correlations = correlation_coefficients[metric].abs().sort_values(ascending=False)
            sorted_correlations = sorted_correlations.drop(metric)  # Exclude the metric itself
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x=sorted_correlations.values, y=sorted_correlations.index)
            plt.title(f'Correlation Coefficients for {metric}')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'correlation_bar_{metric}.png'))
            save_plot(fig, f"correlation_bar_{metric}.png")
            plt.show()
            logging.info(f"Correlation bar plot for {metric} saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting correlation bars: {e}")

def plot_correlation_heatmap(correlation_matrix, output_dir):
    """
    Plot a heatmap for the correlation matrix.

    Args:
    - correlation_matrix (pd.DataFrame): DataFrame containing the correlation matrix.
    - output_dir (str): Directory to save the plot.
    """
    try:
        fig = plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        save_plot(fig, "correlation_heatmap.png")
        plt.show()
        logging.info("Correlation heatmap plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting the correlation heatmap: {e}")
