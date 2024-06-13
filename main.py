import pandas as pd
import logging
import os
from data_preprocessing import load_dataset, clean_data, normalize_data, split_data
from eda import generate_summary_statistics, visualize_data, identify_key_features
from feature_engineering import create_new_features, feature_selection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load the dataset
    file_path = "data/system_metrics.csv"  # Ensure this path is correct
    df = load_dataset(file_path)

    # Clean the data
    df = clean_data(df)

    # Normalize the data
    try:
        df, scaler = normalize_data(df)
    except ValueError as e:
        logger.error(f"Normalization failed: {e}")
        return

    # Split the data
    target_column = "CPU Utilization (%)"  # Replace with your actual target column name
    try:
        X_train, X_test, y_train, y_test = split_data(df, target_column)
    except ValueError as e:
        logger.error(f"Data split failed: {e}")
        return

    # Ensure the preprocessed_data directory exists
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed data if needed
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    logger.info("Preprocessed data saved successfully.")

    # Load preprocessed data for EDA
    X_train = pd.read_csv(f"{output_dir}/X_train.csv")
    X_test = pd.read_csv(f"{output_dir}/X_test.csv")
    y_train = pd.read_csv(f"{output_dir}/y_train.csv")
    y_test = pd.read_csv(f"{output_dir}/y_test.csv")

    # Combine X and y for EDA
    y_train.columns = ["Target"]
    train_data = pd.concat([X_train, y_train], axis=1)

    # Generate summary statistics
    summary = generate_summary_statistics(train_data)

    # Visualize the data
    target_column = "Target"  # Change if the target column has a different name
    visualize_data(train_data, target_column)

    # Identify key features
    key_features = identify_key_features(train_data, target_column)
    logger.info(f"Identified key features:\n{key_features}")

    # Create new features
    X_train = create_new_features(X_train)
    X_test = create_new_features(X_test)

    # Perform feature selection
    feature_importances = feature_selection(X_train, y_train)

    # Save the updated datasets with new features
    X_train.to_csv(f"{output_dir}/X_train_with_new_features.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_with_new_features.csv", index=False)
    logger.info("New features created and data saved successfully.")


if __name__ == "__main__":
    main()
