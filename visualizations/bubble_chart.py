import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_bubble_chart(data):
    """
    Plot a bubble chart for the specified data.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    """
    try:
        fig = plt.figure(figsize=(14, 10))
        scatter = plt.scatter(
            data["CPU Utilization (%)"],
            data["Memory Utilization (%)"],
            s=data["Network I/O Throughput (Mbps)"] / 500,
            c=data["Disk I/O Throughput (MB/s)"],
            alpha=0.6,
            cmap="viridis",
            edgecolor="w",
            linewidth=0.5,
        )
        plt.colorbar(scatter, label="Disk I/O Throughput (MB/s)")
        plt.xlabel("CPU Utilization (%)", fontsize=14)
        plt.ylabel("Memory Utilization (%)", fontsize=14)
        plt.title("Bubble Chart of Chaotic Server Metrics", fontsize=16, weight="bold")
        plt.xlim(20, 120)
        plt.ylim(10, 110)
        plt.grid(True, linestyle="--", linewidth=0.5)
        save_plot(fig, "bubble_chart.png")
        plt.show()
        logging.info("Bubble chart plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting the bubble chart: {e}")
