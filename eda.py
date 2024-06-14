import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_summary_statistics(df):
    """Generate summary statistics of the data."""
    summary = df.describe()
    logger.info(f"Summary statistics:\n{summary}")
    return summary


def visualize_data(df, target_column):
    """Visualize the data to understand distributions, correlations, and trends."""
    # Correlation matrix
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 10},
    )
    # plt.title("Correlation Matrix of VM Performance Metrics", fontsize=16)
    # plt.xticks(rotation=45, ha="right", fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.show()
    # # Distribution of target column
    # plt.figure(figsize=(10, 6))
    # sns.histplot(df[target_column], kde=True)
    # plt.title(f"Distribution of {target_column}")
    # plt.xlabel(f"{target_column} (Standardized Units)")
    # plt.ylabel("Frequency")
    # plt.legend(labels=[f"{target_column} Distribution", "KDE"])
    # plt.show()
    # # Pairplot of features and target
    # plt.figure(figsize=(10, 6))
    # sns.pairplot(df, diag_kind="kde")
    # plt.show()


def identify_key_features(df, target_column):
    """Identify key features impacting VM performance and resource allocation."""
    correlation_matrix = df.corr()
    key_features = correlation_matrix[target_column].sort_values(ascending=False)
    logger.info(f"Key features impacting {target_column}:\n{key_features}")
    return key_features
