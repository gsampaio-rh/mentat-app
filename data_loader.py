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
        operational_data = pd.read_csv(operational_file)
        business_data = pd.read_csv(business_file)
        logging.info("Data loaded successfully.")
        return operational_data, business_data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None, None
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data: {e}")
        return None, None
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error: {e}")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None
