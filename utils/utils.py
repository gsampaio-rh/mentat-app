import logging
import matplotlib.pyplot as plt
import os
from config.config import Config


def save_plot(fig, filename):
    """
    Save the plot to the specified filename.

    Args:
    - fig (matplotlib.figure.Figure): Figure object to save.
    - filename (str): Path to save the figure.
    """
    try:
        output_dir = Config.OUTPUT_DIR
        fig.savefig(os.path.join(output_dir, filename))
        logging.info(f"Plot saved successfully as {os.path.join(output_dir, filename)}")
    except Exception as e:
        logging.error(f"An error occurred while saving the plot: {e}")
