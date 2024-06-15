import pandas as pd
import logging


def handle_missing_values(data):
    """
    Handle missing values in the data.

    Parameters:
    data (pd.DataFrame): The input data with potential missing values.

    Returns:
    pd.DataFrame: Data with missing values handled.
    """
    try:
        # Fill missing values with the mean of the column
        data = data.fillna(data.mean())
        logging.info("Missing values handled successfully.")
        return data
    except Exception as e:
        logging.error(f"An error occurred while handling missing values: {e}")
        return None


def merge_data(operational_data, business_data):
    """
    Merge operational and business data on a common key.

    Parameters:
    operational_data (pd.DataFrame): The operational metrics data.
    business_data (pd.DataFrame): The business metrics data.

    Returns:
    pd.DataFrame: The merged data.
    """
    try:
        merged_data = pd.merge(operational_data, business_data, on="timestamp")
        logging.info("Data merged successfully.")
        return merged_data
    except KeyError as e:
        logging.error(f"Merge key error: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while merging data: {e}")
        return None


def normalize_data(data):
    """
    Normalize the data.

    Parameters:
    data (pd.DataFrame): The input data to normalize.

    Returns:
    pd.DataFrame: The normalized data.
    """
    try:
        normalized_data = (data - data.min()) / (data.max() - data.min())
        logging.info("Data normalized successfully.")
        return normalized_data
    except Exception as e:
        logging.error(f"An error occurred while normalizing data: {e}")
        return None
