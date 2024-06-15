# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


def normalize_data(df, features):
    """
    Normalize the specified features in the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to normalize.

    Returns:
    - pd.DataFrame: DataFrame with normalized features.
    """
    # Initial Data Description
    initial_description = df[features].describe()
    print(f"Initial Data Description:\n{initial_description}")

    # Normalizing Data
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[features] = scaler.fit_transform(df_normalized[features])

    # Final Data Description
    final_description = df_normalized[features].describe()
    print(f"Final Data Description:\n{final_description}")

    return df_normalized


def clean_and_scale_data(data, features):
    """
    Clean and scale the specified features in the DataFrame.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to clean and scale.

    Returns:
    - pd.DataFrame: Cleaned DataFrame
    - np.array: Scaled data
    """
    # Initial Data Shape
    initial_shape = data.shape
    print(f"Initial Data Shape: {initial_shape}")

    # Missing Values before Cleaning
    missing_values_before = data[features].isnull().sum()
    print(f"Missing Values Before Cleaning:\n{missing_values_before}")

    # Cleaning Data
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=features)

    # Rows Dropped
    rows_dropped = initial_shape[0] - data.shape[0]
    print(f"Rows Dropped: {rows_dropped}")

    # Missing Values after Cleaning
    missing_values_after = data[features].isnull().sum()
    print(f"Missing Values After Cleaning:\n{missing_values_after}")

    # Final Data Shape
    final_shape = data.shape
    print(f"Final Data Shape: {final_shape}")

    # Scaling Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    return data, scaled_data


def normalize_profiles(profiles):
    """
    Normalize the profiles DataFrame.

    Args:
    - profiles (pd.DataFrame): DataFrame containing the profiles.

    Returns:
    - pd.DataFrame: Normalized profiles DataFrame.
    """
    scaler = MinMaxScaler()
    normalized_profiles = pd.DataFrame(
        scaler.fit_transform(profiles), columns=profiles.columns, index=profiles.index
    )
    return normalized_profiles


def preprocess_for_utilization(merged_df):
    utilization_df = merged_df[
        ["Server Configuration", "CPU Utilization (%)", "Memory Utilization (%)"]
    ]
    avg_utilization_df = (
        utilization_df.groupby("Server Configuration").mean().reset_index()
    )
    return avg_utilization_df


def preprocess_for_cost_reduction(merged_df):
    cost_df = merged_df[
        [
            "Server Configuration",
            "Operational Costs ($)",
            "Customer Satisfaction (CSAT)",
            "Response Time (ms)",
            "Service Uptime (%)",
        ]
    ]
    avg_cost_df = cost_df.groupby("Server Configuration").mean().reset_index()
    return avg_cost_df


def handle_outliers(df, features, method="IQR"):
    """
    Handle outliers in the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to handle outliers.
    - method (str): Method to handle outliers ('IQR' or 'z-score').

    Returns:
    - pd.DataFrame: DataFrame with outliers handled.
    """
    if method == "IQR":
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    elif method == "z-score":
        from scipy import stats

        df = df[(np.abs(stats.zscore(df[features])) < 3).all(axis=1)]
    return df


def add_new_features(df):
    """
    Add new features to the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the original features.

    Returns:
    - pd.DataFrame: DataFrame with new features added.
    """
    df["CPU_to_Memory_Ratio"] = df["CPU Utilization (%)"] / df["Memory Utilization (%)"]
    df["Disk_to_Network_Throughput_Ratio"] = (
        df["Disk I/O Throughput (MB/s)"] / df["Network I/O Throughput (Mbps)"]
    )
    df["Cost_per_Satisfaction"] = (
        df["Operational Costs ($)"] / df["Customer Satisfaction (CSAT)"]
    )
    df["Response_Time_per_Satisfaction"] = (
        df["Response Time (ms)"] / df["Customer Satisfaction (CSAT)"]
    )

    # Handle potential infinite values due to division by zero
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df
