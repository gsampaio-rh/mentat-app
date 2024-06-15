import logging
import matplotlib.pyplot as plt


def save_plot(fig, filename):
    """
    Save the plot to the specified filename.

    Args:
    - fig (matplotlib.figure.Figure): Figure object to save.
    - filename (str): Path to save the figure.
    """
    try:
        fig.savefig(filename)
        logging.info(f"Plot saved successfully as {filename}")
    except Exception as e:
        logging.error(f"An error occurred while saving the plot: {e}")
