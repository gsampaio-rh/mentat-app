#utils.property

import os

OUTPUT_DIR = "output"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename):
    """
    Save the current plot to the output directory.

    Args:
    - fig (matplotlib.figure.Figure): The figure object to save.
    - filename (str): The name of the file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    print(f"Saved plot to {filepath}")
