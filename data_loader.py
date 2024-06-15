import pandas as pd
import logging


def load_data(operational_file, business_file):
    """
    Load operational and business data from CSV files.

    Parameters:
    operational_file (str): Path to the operational metrics CSV file.
    business_file (str): Path to the business metrics CSV file.

    Returns:
    pd.DataFrame, pd.DataFrame: DataFrames containing the operational and business data.
    """
    try:
        logging.info(f"Attempting to load operational data from {operational_file}")
        operational_data = pd.read_csv(operational_file)
        logging.info(
            f"Operational data loaded successfully with shape {operational_data.shape}"
        )

        logging.info(f"Attempting to load business data from {business_file}")
        business_data = pd.read_csv(business_file)
        logging.info(
            f"Business data loaded successfully with shape {business_data.shape}"
        )

        return operational_data, business_data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}. Error: {e}")
        return None, None
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data in file: {e}")
        return None, None
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error in file: {e}")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data: {e}")
        return None, None
