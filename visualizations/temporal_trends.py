import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_temporal_trends(data, metrics):
    """
    Plot temporal trends for the specified metrics.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - metrics (list): List of metrics to plot.
    """
    try:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
        for i, metric in enumerate(metrics):
            data.plot(x="Timestamp", y=metric, ax=axes[i])
            axes[i].set_title(f"Temporal Trend of {metric}")
        plt.tight_layout()
        save_plot(fig, "temporal_trends.png")
        plt.show()
        logging.info("Temporal trends plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting temporal trends: {e}")
