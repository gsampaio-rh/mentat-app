import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# Configure logging
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


def preprocess_data(df):
    # Select only numeric columns for imputation
    numeric_df = df.select_dtypes(include=[np.number])

    # Handle missing values for numeric columns
    imputer = SimpleImputer(strategy="median")
    numeric_df = pd.DataFrame(
        imputer.fit_transform(numeric_df), columns=numeric_df.columns
    )

    # Power transform to handle skewness
    pt = PowerTransformer()
    numeric_df[numeric_df.columns] = pt.fit_transform(numeric_df[numeric_df.columns])

    # Replace the original numeric columns with the processed ones
    df[numeric_df.columns] = numeric_df

    # Reset index to ensure alignment
    df.reset_index(drop=True, inplace=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering to create new features.
    """
    df["CPU Utilization Ratio"] = df["CPU Usage (seconds)"] / df["CPU Utilization (%)"]
    df["Memory Utilization Ratio"] = (
        df["Memory Usage (bytes)"] / df["Memory Utilization (%)"]
    )
    df["Network I/O Ratio"] = (
        df["Network Receive (bytes)"] + df["Network Transmit (bytes)"]
    ) / df["Network I/O Throughput (Mbps)"]
    df["Storage I/O Ratio"] = (
        df["Storage Read (bytes)"] + df["Storage Write (bytes)"]
    ) / df["Disk I/O Throughput (MB/s)"]
    df["System Load Per Core"] = df["System Load Average"] / 4
    df["Peak CPU Usage"] = (df["CPU Utilization (%)"] > 90).astype(int)
    df["Peak Memory Usage"] = (df["Memory Utilization (%)"] > 90).astype(int)
    df["Peak Network Usage"] = (df["Network I/O Throughput (Mbps)"] > 90).astype(int)
    df["Peak Storage Usage"] = (df["Disk I/O Throughput (MB/s)"] > 90).astype(int)
    logging.info("Feature engineering completed.")
    return df


def check_data_balance(df: pd.DataFrame) -> bool:
    """
    Check if the target variable 'Bottleneck' is balanced.
    """
    bottleneck_counts = df["Bottleneck"].value_counts()
    logging.info(
        "Class distribution in the target variable 'Bottleneck':\n"
        + str(bottleneck_counts)
    )
    return len(bottleneck_counts) > 1


def create_synthetic_bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic bottleneck instances to balance the dataset.
    """
    synthetic_bottlenecks = df.sample(10).copy()
    synthetic_bottlenecks["Peak CPU Usage"] = 1
    synthetic_bottlenecks["Peak Memory Usage"] = 1
    synthetic_bottlenecks["Peak Network Usage"] = 1
    synthetic_bottlenecks["Peak Storage Usage"] = 1
    synthetic_bottlenecks["Bottleneck"] = 1
    logging.info("Synthetic bottlenecks created.")
    return pd.concat([df, synthetic_bottlenecks], ignore_index=True)


def balance_dataset(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Balance the dataset using SMOTE.
    """
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logging.info("Dataset balanced using SMOTE.")
    return X_resampled, y_resampled


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Tune the Random Forest model using RandomizedSearchCV.
    """
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_features": ["sqrt", "log2"],
        "max_depth": [10, 20, 30, 40, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    logging.info("Model tuning completed.")
    return random_search.best_estimator_


def train_bottleneck_model(df: pd.DataFrame) -> tuple:
    """
    Train the Random Forest model to predict bottlenecks.
    """
    df["Bottleneck"] = df[
        [
            "Peak CPU Usage",
            "Peak Memory Usage",
            "Peak Network Usage",
            "Peak Storage Usage",
        ]
    ].max(axis=1)
    feature_cols = [
        "CPU Utilization Ratio",
        "Memory Utilization Ratio",
        "Network I/O Ratio",
        "Storage I/O Ratio",
        "System Load Per Core",
    ]
    X = df[feature_cols]
    y = df["Bottleneck"]

    if not check_data_balance(df):
        logging.warning(
            "Only one class present in the target variable. Creating synthetic bottlenecks."
        )
        df = create_synthetic_bottlenecks(df)
        X = df[feature_cols]
        y = df["Bottleneck"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test_indices = X_test.index
    X_train, y_train = balance_dataset(X_train, y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    best_rf = tune_model(X_train, y_train)
    y_pred = best_rf.predict(X_test)

    if len(np.unique(y_test)) == 1:
        logging.warning(
            "Only one class present in y_test. Skipping ROC AUC, Precision, Recall, and F1 scores."
        )
        roc_auc = precision = recall = f1 = "N/A"
    else:
        roc_auc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info(f"ROC AUC Score: {roc_auc}")

    return best_rf, feature_cols, X_test, y_test, y_pred, X_test_indices


def plot_ellipses(gmm, ax):
    for n, color in enumerate(["r", "g", "b"]):
        if n >= len(gmm.means_):
            continue
        covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = plt.matplotlib.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=angle, color=color, alpha=0.3
        )
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)


def plot_classification_results_with_clusters(
    X_test, y_test, y_pred, df, X_test_indices
):
    if "Number of Servers" not in df.columns:
        print(
            "Warning: 'Number of Servers' column not found. Adding placeholder values."
        )
        df["Number of Servers"] = np.random.randint(1, 100, size=len(df))
    else:
        if (
            df["Number of Servers"].isnull().any()
            or (df["Number of Servers"] <= 0).any()
        ):
            print(
                "Warning: 'Number of Servers' column has invalid values. Replacing with placeholder values."
            )
            df["Number of Servers"] = np.random.randint(1, 100, size=len(df))

    # Ensure the indices are aligned correctly
    try:
        df_aligned = df.loc[X_test_indices]
    except KeyError as e:
        missing_indices = list(set(X_test_indices) - set(df.index))
        print(
            f"Warning: The following indices are missing and will be skipped: {missing_indices}"
        )
        df_aligned = df.loc[df.index.intersection(X_test_indices)]
        X_test = X_test[np.isin(X_test_indices, df.index)]
        y_pred = y_pred[np.isin(X_test_indices, df.index)]
        X_test_indices = X_test_indices[np.isin(X_test_indices, df.index)]

    n_components = min(len(X_test), 3)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_test[:, :2])

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sizes = df_aligned["Number of Servers"] * 10
    scatter = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pred,
        s=sizes,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="w",
        label=y_pred,
    )
    plot_ellipses(gmm, ax)
    plt.xlabel("CPU Utilization Ratio")
    plt.ylabel("Memory Utilization Ratio")
    plt.title("Random Forest Classification Results with Server Counts and Clusters")

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend1 = ax.legend(handles, labels, loc="upper right", title="Number of Servers")
    ax.add_artist(legend1)

    handles, labels = scatter.legend_elements()
    legend2 = ax.legend(
        handles, ["Non-Bottleneck", "Bottleneck"], loc="upper left", title="Prediction"
    )
    ax.add_artist(legend2)

    plt.show()

    plt.show()


def visualize_bottlenecks(df: pd.DataFrame, model: RandomForestClassifier):
    """
    Visualize the predicted bottlenecks.
    """
    feature_cols = [
        "CPU Utilization Ratio",
        "Memory Utilization Ratio",
        "Network I/O Ratio",
        "Storage I/O Ratio",
        "System Load Per Core",
    ]
    df["Predicted Bottleneck"] = model.predict(df[feature_cols])
    bottleneck_counts = df["Predicted Bottleneck"].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=bottleneck_counts.index, y=bottleneck_counts.values, palette="viridis"
    )
    plt.title("Predicted Resource Bottlenecks", fontsize=16)
    plt.xlabel("Bottleneck", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    for index, value in enumerate(bottleneck_counts.values):
        plt.text(index, value + 10, str(value), ha="center", fontsize=12)

    plt.xticks([0, 1], ["No Bottleneck", "Bottleneck"], fontsize=12)

    explanation_text = (
        "Bottleneck (1): A predicted resource constraint\n"
        "that could impact system performance. If a bottleneck occurs,\n"
        "users may experience slower response times or interruptions.\n"
        "No Bottleneck (0): System is operating normally without\n"
        "resource constraints, ensuring smooth and efficient performance."
    )
    plt.gcf().text(
        0.75,
        0.9,
        explanation_text,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8),
        ha="left",
    )
    plt.show()


def plot_feature_importance(model: RandomForestClassifier, feature_cols: list):
    """
    Plot the feature importance of the model.
    """
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
        feature_importance = feature_importance.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=feature_importance.index)
        plt.title("Feature Importance in Bottleneck Prediction")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    else:
        logging.error("Model does not have feature_importances_ attribute.")


def main():
    filepath = os.path.join("data", "real_faang.csv")
    df = load_data(filepath)
    df = preprocess_data(df)
    df = feature_engineering(df)
    model, feature_cols, X_test, y_test, y_pred, X_test_indices = (
        train_bottleneck_model(df)
    )

    if model is None:
        logging.error(
            "Model training skipped due to lack of class diversity in the target variable."
        )
        return

    plot_classification_results_with_clusters(
        X_test, y_test, y_pred, df, X_test_indices
    )
    visualize_bottlenecks(df, model)
    plot_feature_importance(model, feature_cols)


if __name__ == "__main__":
    main()
