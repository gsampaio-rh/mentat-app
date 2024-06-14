import pandas as pd
import logging
import os
from data_preprocessing import load_dataset, clean_data, normalize_data, split_data
from eda import generate_summary_statistics, visualize_data, identify_key_features
from feature_engineering import create_new_features, feature_selection
from model_selection import (
    linear_regression,
    random_forest,
    random_forest_with_hyperparameter_tuning,
    xgboost_regressor,
    xgboost_with_hyperparameter_tuning,
    k_means_clustering,
)
from visualization import (
    plot_model_performance,
    plot_feature_importances,
    plot_residuals,
)
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_missing_values(df):
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns

    # Impute numeric columns with mean strategy
    imputer_numeric = SimpleImputer(strategy="mean")
    df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

    # Impute non-numeric columns with most frequent strategy (if any non-numeric columns exist)
    if len(non_numeric_columns) > 0:
        imputer_non_numeric = SimpleImputer(strategy="most_frequent")
        df[non_numeric_columns] = imputer_non_numeric.fit_transform(
            df[non_numeric_columns]
        )

    return df


def main():
    # Load the dataset
    file_path = "data/system_metrics.csv"  # Ensure this path is correct
    df = load_dataset(file_path)

    # Clean the data
    df = clean_data(df)

    # Drop non-numeric columns
    df = df.drop(columns=["timestamp", "Server Name"])

    # Handle missing values
    df = handle_missing_values(df)

    # Ensure no remaining NaNs
    if df.isnull().sum().sum() > 0:
        logger.error(
            "There are still missing values in the dataset after handling missing values."
        )
        return

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

    # Define features and target for regression models
    features = X_train.columns.tolist()
    target = "Target"

    # Train and evaluate the Random Forest model
    rf_model, rf_mse, rf_r2 = random_forest(X_train, y_train, features, target)
    logger.info(f"Random Forest - MSE: {rf_mse}, R2: {rf_r2}")

    # Visualize Random Forest Performance
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    plot_model_performance(y_train, y_pred_train, y_test, y_pred_test, "Random Forest")
    plot_feature_importances(rf_model, features, "Random Forest")
    plot_residuals(y_test, y_pred_test, "Random Forest")

    # Hyperparameter Tuning for Random Forest
    logger.info("Starting hyperparameter tuning for Random Forest.")
    best_rf_model, rf_mse, rf_r2 = random_forest_with_hyperparameter_tuning(
        X_train, y_train, features, target
    )
    logger.info(f"Best Random Forest Model: {best_rf_model}")
    logger.info(f"Best Random Forest - MSE: {rf_mse}, R2: {rf_r2}")

    # Train and evaluate the XGBoost model
    # xgb_model, xgb_mse, xgb_r2 = xgboost_regressor(X_train, y_train, features, target)
    # logger.info(f"XGBoost - MSE: {xgb_mse}, R2: {xgb_r2}")

    # # Visualize XGBoost Performance
    # y_pred_train = xgb_model.predict(X_train)
    # y_pred_test = xgb_model.predict(X_test)
    # plot_model_performance(y_train, y_pred_train, y_test, y_pred_test, "XGBoost")
    # plot_feature_importances(xgb_model, features, "XGBoost")
    # plot_residuals(y_test, y_pred_test, "XGBoost")

    # # Hyperparameter Tuning for XGBoost
    # best_xgb_model, xgb_mse, xgb_r2 = xgboost_with_hyperparameter_tuning(
    #     X_train, y_train, features, target
    # )
    # logger.info(f"Best XGBoost Model: {best_xgb_model}")
    # logger.info(f"Best XGBoost - MSE: {xgb_mse}, R2: {xgb_r2}")


if __name__ == "__main__":
    main()
