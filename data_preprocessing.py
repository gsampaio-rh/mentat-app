# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_data(df, features):
    """
    Normalize the specified features in the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to normalize.

    Returns:
    - pd.DataFrame: DataFrame with normalized features.
    """
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[features] = scaler.fit_transform(df_normalized[features])
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
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=features)
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
