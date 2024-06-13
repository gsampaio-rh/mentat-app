import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(file_path):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    logger.info(f"Initial dataset shape: {df.shape}")
    # Handle missing values
    missing_values_count = df.isnull().sum().sum()
    logger.info(f"Missing values in dataset: {missing_values_count}")
    df = df.dropna()  # Simple approach: drop rows with missing values
    logger.info(f"Dataset shape after dropping missing values: {df.shape}")
    # Handle outliers using IQR method
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    df = df[
        ~(
            (df[numeric_cols] < (Q1 - 1.5 * IQR))
            | (df[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)
    ]
    logger.info(f"Dataset shape after removing outliers: {df.shape}")
    return df


def normalize_data(df):
    """Normalize or standardize the dataset."""
    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric features to be scaled: {list(numeric_features)}")
    if df[numeric_features].shape[0] == 0:
        raise ValueError("No samples left after cleaning to normalize.")
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df, scaler


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test
