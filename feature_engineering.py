import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_new_features(df):
    """Create new features based on domain knowledge."""
    df["CPU_to_Memory"] = df["CPU (GHz)"] / df["RAM Memory (GiB)"]
    df["Total_Network_IO"] = (
        df["Network Receive (bytes)"] + df["Network Transmit (bytes)"]
    )
    df["Total_Storage_IO"] = df["Storage Read (bytes)"] + df["Storage Write (bytes)"]
    return df


def handle_non_numeric(df):
    """Handle non-numeric columns, such as datetime."""
    # Convert datetime columns to numerical values if present
    for col in df.select_dtypes(include=["datetime", "object"]).columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].apply(lambda x: x.timestamp())
        except Exception:
            df.drop(columns=[col], inplace=True)
            logger.info(f"Dropped non-numeric column: {col}")
    return df


def feature_selection(X_train, y_train):
    """Perform feature selection to identify the most relevant features for the model."""
    # Handle non-numeric columns
    X_train = handle_non_numeric(X_train)

    # Remove the 'timestamp' column if it exists
    if "timestamp" in X_train.columns:
        X_train = X_train.drop(columns=["timestamp"])
        logger.info("Dropped 'timestamp' column for feature selection.")

    # Use a RandomForestRegressor to compute feature importances
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train.values.ravel())

    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )
    feature_importances = feature_importances.sort_values(
        by="Importance", ascending=False
    )

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=feature_importances)
    plt.title("Feature Importances")
    plt.show()

    logger.info(f"Feature importances:\n{feature_importances}")
    return feature_importances
