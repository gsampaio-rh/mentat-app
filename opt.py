import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Key Metrics
key_metrics = {
    "CPU Utilization (%)": "CPU Usage",
    "Memory Utilization (%)": "Memory Usage",
    "Disk I/O Throughput (MB/s)": "Disk Throughput",
    "Network I/O Throughput (Mbps)": "Network Throughput",
}

additional_metrics = [
    "System Load Average",
    "Pod CPU Utilization (%)",
    "Pod Memory Utilization (%)",
    "Cluster CPU Utilization (%)",
    "Cluster Memory Utilization (%)",
]


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


def plot_all_metrics_single_chart(df: pd.DataFrame):
    """
    Plot all key metrics in a single chart with different colors.
    """
    plt.figure(figsize=(12, 8))

    for metric, label in key_metrics.items():
        plt.scatter(df.index, df[metric], label=label, s=10)

    for metric in additional_metrics:
        plt.scatter(df.index, df[metric], label=metric, s=10)

    plt.title("All Key Metrics")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_combined_metrics(df: pd.DataFrame):
    """
    Plot combined histograms and KDE plots for key metrics with explicit legends.
    """

    plt.figure(figsize=(12, 8))
    for i, (metric, label) in enumerate(key_metrics.items(), 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[metric], kde=True, label="KDE", color="blue")
        plt.title(f"Distribution of {label}")
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_boxplots(df: pd.DataFrame):
    """
    Plot box plots for key metrics to identify outliers.
    """

    plt.figure(figsize=(12, 8))
    for i, (metric, label) in enumerate(key_metrics.items(), 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[metric], color="blue")
        plt.title(f"Box plot of {label}")
        plt.xlabel(label)
    plt.tight_layout()
    plt.show()


def generate_summary_statistics(df: pd.DataFrame):
    """
    Generate and log summary statistics for key metrics.
    """
    summary_stats = df[list(key_metrics.keys()) + additional_metrics].describe()
    logging.info(f"Summary statistics for key metrics:\n{summary_stats}")


def perform_eda(df: pd.DataFrame):
    """
    Perform exploratory data analysis on the dataframe.
    """
    # Plot combined metrics
    plot_combined_metrics(df)
    # Plot box plots
    plot_boxplots(df)
    # Generate summary statistics
    generate_summary_statistics(df)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the dataframe.
    """
    # Example: Create interaction term between CPU and Memory Utilization
    df["CPU_Memory_Interaction"] = (
        df["CPU Utilization (%)"] * df["Memory Utilization (%)"]
    )
    logging.info("Feature engineering completed.")
    return df


def main():

    filepath = os.path.join("data", "real_faang.csv")
    df = load_data(filepath)
    df = preprocess_data(df)

    # Initial combined metrics plot
    plot_all_metrics_single_chart(df)

    # Perform EDA
    perform_eda(df)

    # Feature engineering
    df = feature_engineering(df)

    # Display first few rows of the preprocessed data for verification
    logging.info(f"First few rows of the preprocessed data:\n{df.head()}")


if __name__ == "__main__":
    main()
