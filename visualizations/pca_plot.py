import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_pca_loadings(loadings):
    """
    Plot PCA loadings.

    Args:
    - loadings (pd.DataFrame): DataFrame containing the PCA loadings.
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        loadings.plot(kind="bar", ax=ax)
        ax.set_title(
            "PCA Loadings for Principal Components 1 and 2", fontsize=16, weight="bold"
        )
        ax.set_xlabel("Features", fontsize=14)
        ax.set_ylabel("Contribution to Principal Component", fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        save_plot(fig, "pca_loadings.png")
        plt.show()
        logging.info("PCA loadings plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting PCA loadings: {e}")
