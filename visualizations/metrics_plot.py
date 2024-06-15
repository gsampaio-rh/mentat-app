import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_all_metrics_single_chart(df, features):
    """
    Plot all key metrics in a single chart with different colors and markers.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    """
    try:
        fig = plt.figure(figsize=(14, 8))
        colors = ["red", "green", "blue", "purple"]
        markers = ["o", "s", "D", "X"]

        for i, metric in enumerate(features):
            plt.scatter(
                df.index, df[metric], label=metric, color=colors[i], marker=markers[i], s=10
            )

        plt.title("All Key Metrics (Normalized)", fontsize=16, weight="bold")
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Normalized Value", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        save_plot(fig, "all_metrics_single_chart.png")
        plt.show()
        logging.info("All key metrics plotted successfully in a single chart.")
    except Exception as e:
        logging.error(f"An error occurred while plotting all metrics in a single chart: {e}")
