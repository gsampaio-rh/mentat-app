import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import save_plot
import logging

def plot_summary_statistics(data, features):
    """
    Plot summary statistics for the specified features.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot summary statistics for.
    """
    try:
        num_features = len(features)
        num_cols = 2  # Number of columns
        num_rows = (num_features + 1) // num_cols  # Calculate number of rows

        # Define colors for each feature
        colors = sns.color_palette("husl", num_features)

        # Adjust the figure size to fit a MacBook 15" screen
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 1.1))

        for i, feature in enumerate(features):
            row, col = divmod(i, num_cols)
            sns.histplot(data[feature], kde=True, ax=axes[row, col], color=colors[i])
            axes[row, col].set_title(f"{feature} Distribution")
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel("Count")
            axes[row, col].grid(True, linestyle="--", linewidth=0.5)

        # Remove any empty subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes.flat[j])

        # Adjust layout
        plt.tight_layout()
        save_plot(fig, "summary_statistics.png")
        plt.show()
        logging.info("Summary statistics plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting summary statistics: {e}")
