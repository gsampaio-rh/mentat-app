# data_loader.py

import pandas as pd


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully read the file: {file_path}")
        return data
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return None
