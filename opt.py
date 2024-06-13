import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from the given file path.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {filepath}")
        raise e


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and transforming skewed features.
    """
    # Handle missing values
    numeric_df = df.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median")
    numeric_df = pd.DataFrame(
        imputer.fit_transform(numeric_df), columns=numeric_df.columns
    )

    # Normalize/standardize the data
    pt = PowerTransformer()
    numeric_df[numeric_df.columns] = pt.fit_transform(numeric_df[numeric_df.columns])
    df[numeric_df.columns] = numeric_df
    df.reset_index(drop=True, inplace=True)
    logging.info("Data preprocessing completed.")
    return df


def plot_histograms(df: pd.DataFrame):
    """
    Plot histograms and KDE plots for key metrics.
    """
    key_metrics = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)",
    ]
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[metric], kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.show()


def plot_boxplots(df: pd.DataFrame):
    """
    Plot box plots for key metrics to identify outliers.
    """
    key_metrics = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)",
    ]
    for metric in key_metrics:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[metric])
        plt.title(f"Box plot of {metric}")
        plt.xlabel(metric)
        plt.show()


def generate_summary_statistics(df: pd.DataFrame):
    """
    Generate and log summary statistics for key metrics.
    """
    key_metrics = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)",
    ]
    summary_stats = df[key_metrics].describe()
    logging.info(f"Summary statistics for key metrics:\n{summary_stats}")


def main():
    filepath = os.path.join("data", "real_faang.csv")
    df = load_data(filepath)
    df = preprocess_data(df)

    # Exploratory Data Analysis (EDA)
    plot_histograms(df)
    plot_boxplots(df)
    generate_summary_statistics(df)

    # Display first few rows of the preprocessed data for verification
    logging.info(f"First few rows of the preprocessed data:\n{df.head()}")


if __name__ == "__main__":
    main()
