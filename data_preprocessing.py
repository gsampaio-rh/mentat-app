import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess_data(operational_data, business_data):
    """
    Preprocess operational and business data.

    Args:
    - operational_data (pd.DataFrame): DataFrame containing the operational data.
    - business_data (pd.DataFrame): DataFrame containing the business data.

    Returns:
    - np.array: Scaled operational data.
    - np.array: Scaled business data.
    """
    try:
        # Standardize the operational data
        scaler = StandardScaler()
        scaled_operational_data = scaler.fit_transform(operational_data)

        # Standardize the business data
        scaled_business_data = scaler.fit_transform(business_data)

        logging.info("Data preprocessing completed successfully.")
        return scaled_operational_data, scaled_business_data
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}")
        return None, None


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
