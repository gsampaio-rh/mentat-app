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

def plot_highlighted_metric(df: pd.DataFrame, metric: str, label: str):
    average_value = df[metric].mean()
    x_mean = len(df) / 2  # Use the middle of the dataset for the annotation
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[metric], "bo", markersize=4)
    plt.axhline(y=average_value, color="r", linestyle="--")
    plt.scatter(x_mean, average_value, color="red", s=100, edgecolor="black", zorder=5)
    plt.title(f"Average {label}")
    plt.xlabel("Index")
    plt.ylabel(label)
    plt.grid(True)
    plt.annotate(
        "Average Value",
        xy=(x_mean, average_value),
        xytext=(x_mean + 10, average_value + 0.2),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.show()

def plot_combined_metrics(df: pd.DataFrame):
    """
    Plot combined histograms and KDE plots for key metrics with explicit legends.
    """
    key_metrics = {
        "CPU Utilization (%)": "CPU Usage",
        "Memory Utilization (%)": "Memory Usage",
        "Disk I/O Throughput (MB/s)": "Disk Throughput",
        "Network I/O Throughput (Mbps)": "Network Throughput",
    }

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
    key_metrics = {
        "CPU Utilization (%)": "CPU Usage",
        "Memory Utilization (%)": "Memory Usage",
        "Disk I/O Throughput (MB/s)": "Disk Throughput",
        "Network I/O Throughput (Mbps)": "Network Throughput",
    }

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
    plot_combined_metrics(df)
    plot_boxplots(df)
    generate_summary_statistics(df)

    # Highlight key metric
    plot_highlighted_metric(df, "CPU Utilization (%)", "CPU Usage")

    # Display first few rows of the preprocessed data for verification
    logging.info(f"First few rows of the preprocessed data:\n{df.head()}")


if __name__ == "__main__":
    main()
