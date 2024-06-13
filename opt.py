import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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

# Example weights (adjust based on importance)
weights = {
    "CPU Utilization (%)": 0.2,
    "Memory Utilization (%)": 0.2,
    "Disk I/O Throughput (MB/s)": 0.15,
    "Network I/O Throughput (Mbps)": 0.15,
    "System Load Average": 0.1,
    "Pod CPU Utilization (%)": 0.1,
    "Pod Memory Utilization (%)": 0.05,
    "Cluster CPU Utilization (%)": 0.05,
    "Cluster Memory Utilization (%)": 0.05,
}

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


def split_data(df: pd.DataFrame, target_column: str):
    """
    Split the data into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logging.info("Data splitting completed.")
    return X_train, X_test, y_train, y_test


def handle_imbalanced_data_regression(X_train, y_train):
    """
    Handle imbalanced data for regression using oversampling.
    """
    # Combine X_train and y_train into a single DataFrame
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate the majority and minority classes
    majority = train_data[
        train_data["Resource_Utilization_Efficiency"]
        > train_data["Resource_Utilization_Efficiency"].median()
    ]
    minority = train_data[
        train_data["Resource_Utilization_Efficiency"]
        <= train_data["Resource_Utilization_Efficiency"].median()
    ]

    # Upsample the minority class
    minority_upsampled = resample(
        minority,
        replace=True,  # sample with replacement
        n_samples=len(majority),  # to match majority class
        random_state=42,
    )  # reproducible results

    # Combine majority and upsampled minority
    upsampled = pd.concat([majority, minority_upsampled])

    # Separate X and y again
    X_resampled = upsampled.drop(columns=["Resource_Utilization_Efficiency"])
    y_resampled = upsampled["Resource_Utilization_Efficiency"]

    logging.info("Handling imbalanced data using oversampling completed.")
    return X_resampled, y_resampled


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
    # Create new features
    df["Resource_Utilization_Efficiency"] = (
        df["CPU Utilization (%)"] * weights["CPU Utilization (%)"]
        + df["Memory Utilization (%)"] * weights["Memory Utilization (%)"]
        + df["Disk I/O Throughput (MB/s)"] * weights["Disk I/O Throughput (MB/s)"]
        + df["Network I/O Throughput (Mbps)"] * weights["Network I/O Throughput (Mbps)"]
        + df["System Load Average"] * weights["System Load Average"]
        + df["Pod CPU Utilization (%)"] * weights["Pod CPU Utilization (%)"]
        + df["Pod Memory Utilization (%)"] * weights["Pod Memory Utilization (%)"]
        + df["Cluster CPU Utilization (%)"] * weights["Cluster CPU Utilization (%)"]
        + df["Cluster Memory Utilization (%)"]
        * weights["Cluster Memory Utilization (%)"]
    ) / sum(
        weights.values()
    )  # Normalize by sum of weights

    df["CPU_Memory_Ratio"] = df["CPU Utilization (%)"] / (
        df["Memory Utilization (%)"] + 1e-5
    )

    # Additional metrics
    df["System_Load_Average_Ratio"] = df["System Load Average"] / (
        df["CPU Utilization (%)"] + 1e-5
    )
    df["Pod_CPU_Memory_Ratio"] = df["Pod CPU Utilization (%)"] / (
        df["Pod Memory Utilization (%)"] + 1e-5
    )
    df["Cluster_CPU_Memory_Ratio"] = df["Cluster CPU Utilization (%)"] / (
        df["Cluster Memory Utilization (%)"] + 1e-5
    )

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(
        df[
            [
                "CPU Utilization (%)",
                "Memory Utilization (%)",
                "Disk I/O Throughput (MB/s)",
                "Network I/O Throughput (Mbps)",
                "System Load Average",
                "Pod CPU Utilization (%)",
                "Pod Memory Utilization (%)",
                "Cluster CPU Utilization (%)",
                "Cluster Memory Utilization (%)",
            ]
        ]
    )
    poly_df = pd.DataFrame(
        poly_features,
        columns=poly.get_feature_names_out(
            [
                "CPU Utilization (%)",
                "Memory Utilization (%)",
                "Disk I/O Throughput (MB/s)",
                "Network I/O Throughput (Mbps)",
                "System Load Average",
                "Pod CPU Utilization (%)",
                "Pod Memory Utilization (%)",
                "Cluster CPU Utilization (%)",
                "Cluster Memory Utilization (%)",
            ]
        ),
    )
    df = pd.concat([df, poly_df], axis=1)

    logging.info("Feature engineering completed.")
    return df


def visualize_insights(df: pd.DataFrame) -> None:
    """
    Visualize key insights from the dataframe.
    """
    fig, axs = plt.subplots(4, figsize=(12, 8))

    required_columns = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Pod_CPU_Memory_Ratio",
        "Cluster_CPU_Memory_Ratio",
        "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Plotting
    try:
        # Figure 1: Resource Utilization Efficiency Over Time
        axs[0].plot(df.index, df["Resource_Utilization_Efficiency"])
        axs[0].set_title("Resource Utilization Efficiency Over Time")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Efficiency")

        # Figure 1: CPU and Memory Utilization Over Time
        axs[1].plot(df.index, df["CPU Utilization (%)"], label="CPU Utilization")
        axs[1].plot(df.index, df["Memory Utilization (%)"], label="Memory Utilization")
        axs[1].set_title("CPU and Memory Utilization Over Time")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Utilization (%)")
        axs[1].legend()

        # Figure 3: Pod and Cluster CPU-Memory Ratios
        axs[2].scatter(df["Pod_CPU_Memory_Ratio"], df["Cluster_CPU_Memory_Ratio"])
        axs[2].set_title("Pod vs. Cluster CPU-Memory Ratios")
        axs[2].set_xlabel("Pod CPU-Memory Ratio")
        axs[2].set_ylabel("Cluster CPU-Memory Ratio")

        # Figure 4: Disk I/O and Network I/O Throughput
        bar_width = 0.35
        disk_io_mean = df["Disk I/O Throughput (MB/s)"].mean()
        network_io_mean = df["Network I/O Throughput (Mbps)"].mean()
        axs[3].bar(
            ["Disk I/O Throughput"],
            disk_io_mean,
            bar_width,
            label="Disk I/O Throughput",
        )
        axs[3].bar(
            ["Network I/O Throughput"],
            network_io_mean,
            bar_width,
            label="Network I/O Throughput",
        )
        axs[3].set_title("Disk I/O and Network I/O Throughput")
        axs[3].set_xlabel("Throughput Type")
        axs[3].set_ylabel("Average Throughput")
        axs[3].legend()

        # Figure 5: System Load Average and CPU Utilization Ratio
        # axs[4].scatter(df["CPU Utilization (%)"], df["System_Load_Average_Ratio"])
        # axs[4].set_title("System Load Average vs. CPU Utilization Ratio")
        # axs[4].set_xlabel("CPU Utilization (%)")
        # axs[4].set_ylabel("System Load Average Ratio")

    except KeyError as e:
        print(f"Key error: {e} - Check if the DataFrame contains the required columns.")
    except Exception as e:
        print(f"An error occurred: {e}")

    plt.tight_layout()
    plt.show()


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

    # Data Splitting
    target_column = "Resource_Utilization_Efficiency"  # Using Resource Utilization Efficiency as the target column
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Handling Imbalanced Data
    X_train_balanced, y_train_balanced = handle_imbalanced_data_regression(
        X_train, y_train
    )

    visualize_insights(df)

    # Display first few rows of the preprocessed data for verification
    logging.info(f"First few rows of the preprocessed data:\n{df.head()}")


if __name__ == "__main__":
    main()
