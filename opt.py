import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging
from sklearn.utils import resample
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import KMeans

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Key Metrics
key_metrics = {
    "CPU Utilization (%)": "CPU Usage",
    "Memory Utilization (%)": "Memory Usage",
    "Network I/O Throughput (Mbps)": "Network Throughput",
}

# Example weights (adjust based on importance)
weights = {
    "CPU Utilization (%)": 0.4,
    "Memory Utilization (%)": 0.4,
    "Network I/O Throughput (Mbps)": 0.2,
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
    numeric_df = df.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median")
    numeric_df = pd.DataFrame(
        imputer.fit_transform(numeric_df), columns=numeric_df.columns
    )

    # Encode categorical features
    categorical_df = df.select_dtypes(include=[object])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_categorical_df = pd.DataFrame(
        encoder.fit_transform(categorical_df),
        columns=encoder.get_feature_names_out(categorical_df.columns),
    )

    # Combine numeric and encoded categorical data
    df_processed = pd.concat([numeric_df, encoded_categorical_df], axis=1)

    # Normalize/standardize the data
    pt = PowerTransformer()
    df_processed[df_processed.columns] = pt.fit_transform(
        df_processed[df_processed.columns]
    )
    df_processed.reset_index(drop=True, inplace=True)

    logging.info("Data preprocessing completed.")
    return df_processed


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
    summary_stats = df[list(key_metrics.keys())].describe()
    logging.info(f"Summary statistics for key metrics:\n{summary_stats}")


def analyze_resource_utilization(df: pd.DataFrame):
    """
    Analyze resource utilization to identify potential bottlenecks.
    """
    resource_columns = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network I/O Throughput (Mbps)",
    ]

    # Identify potential bottlenecks by checking high utilization
    high_cpu = df["CPU Utilization (%)"] > 90
    high_memory = df["Memory Utilization (%)"] > 90
    high_network_io = df["Network I/O Throughput (Mbps)"] > 90

    bottlenecks = high_cpu | high_memory | high_network_io

    logging.info(f"Number of potential bottlenecks: {bottlenecks.sum()}")
    logging.info(f"High CPU Utilization: {high_cpu.sum()}")
    logging.info(f"High Memory Utilization: {high_memory.sum()}")
    logging.info(f"High Network I/O Throughput: {high_network_io.sum()}")

    return df[bottlenecks], resource_columns


def visualize_resource_utilization(df: pd.DataFrame, resource_columns: list):
    """
    Visualize resource utilization to identify bottlenecks.
    """
    plt.figure(figsize=(12, 8))

    for column in resource_columns:
        sns.histplot(df[column], kde=True, label=column)

    plt.title("Resource Utilization Analysis")
    plt.xlabel("Utilization")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def perform_clustering_analysis(df: pd.DataFrame, n_clusters: int = 3):
    """
    Perform clustering analysis on the dataset to group VMs based on resource usage patterns.
    """
    resource_columns = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network I/O Throughput (Mbps)",
    ]

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df[resource_columns])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(standardized_data)

    logging.info(f"Cluster centers: {kmeans.cluster_centers_}")

    return df, kmeans.cluster_centers_


def visualize_clusters_with_distribution(df: pd.DataFrame, cluster_centers):
    """
    Visualize overutilized, underutilized, and efficient VMs in one image along with the cluster distribution.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Scatter plot of clusters
    sns.scatterplot(
        x="CPU Utilization (%)",
        y="Memory Utilization (%)",
        hue="Cluster",
        data=df,
        palette="viridis",
        alpha=0.6,
        edgecolor="w",
        ax=axs[0],
    )

    axs[0].scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        s=300,
        c="red",
        label="Cluster Centers",
        edgecolors="black",
    )

    axs[0].set_title(
        "Clustering Analysis of VMs: Overutilized, Underutilized, and Efficient VMs"
    )
    axs[0].set_xlabel("CPU Utilization (%)")
    axs[0].set_ylabel("Memory Utilization (%)")
    axs[0].legend(title="Cluster")
    axs[0].grid(True)

    # Bar plot of cluster distribution
    cluster_counts = df["Cluster"].value_counts().sort_index()
    sns.barplot(
        x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=axs[1]
    )
    axs[1].set_title("Number of Servers in Each Cluster")
    axs[1].set_xlabel("Cluster")
    axs[1].set_ylabel("Number of Servers")
    axs[1].grid(True)

    plt.show()

    logging.info(f"Cluster counts: {cluster_counts.to_dict()}")


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
    summary_stats = df[list(key_metrics.keys())].describe()
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
    Perform feature engineering to create new features.
    """
    df["Resource_Utilization_Efficiency"] = (
        df["CPU Utilization (%)"] * weights["CPU Utilization (%)"]
        + df["Memory Utilization (%)"] * weights["Memory Utilization (%)"]
        + df["Network I/O Throughput (Mbps)"] * weights["Network I/O Throughput (Mbps)"]
    ) / sum(
        weights.values()
    )  # Normalize by sum of weights

    df["CPU_Memory_Ratio"] = df["CPU Utilization (%)"] / (
        df["Memory Utilization (%)"] + 1e-5
    )

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

    except KeyError as e:
        logging.error(
            f"Key error: {e} - Check if the DataFrame contains the required columns."
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    plt.tight_layout()
    plt.show()


def train_baseline_model(X_train, y_train):
    """
    Train a baseline Random Forest model.
    """
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    logging.info("Baseline model training completed.")
    return rf


def hyperparameter_tuning(X_train, y_train):
    """
    Tune hyperparameters using RandomizedSearchCV.
    """
    rf = RandomForestRegressor(random_state=42)

    # Define the parameter grid
    param_dist = {
        "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        "max_features": ["sqrt", "log2", None],
        "max_depth": [int(x) for x in np.linspace(10, 110, num=11)],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }

    # Randomized search
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_random.fit(X_train, y_train)

    logging.info("Hyperparameter tuning completed.")
    return rf_random.best_estimator_


def plot_model_performance(model, X_test, y_test):
    """
    Plot the model performance on the test set with additional details.
    """
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.grid(True)

    # Adding mean squared error and R2 score to the plot
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plt.text(
        0.05,
        0.95,
        f"MSE: {mse:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    plt.text(
        0.05,
        0.90,
        f"R2 Score: {r2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    # Adding a legend
    plt.legend(["Data points"], loc="best")

    # Explanation for non-technical people
    explanation_text = (
        "MSE (Mean Squared Error) is a measure of how well the model's predictions match the actual values. "
        "A lower MSE indicates better accuracy.\n"
        "R2 Score (R-squared) indicates the proportion of variance in the dependent variable that is predictable from the independent variables. "
        "A higher R2 Score indicates a better fit."
    )
    plt.gcf().text(
        0.05,
        0.85,
        explanation_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
        ha="left",
    )

    plt.show()

    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R2 Score: {r2}")


def plot_residuals(y_test, y_pred):
    """
    Plot the residuals of the model.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True)
    plt.show()


def plot_feature_importances(model, feature_names, top_n=10):
    """
    Plot the feature importances of the model, focusing on the top N features.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_feature_names = np.array(feature_names)[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_feature_names, palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    logging.info(f"Top {top_n} feature importances plotted.")


def validate_model(model, X_train, y_train):
    """
    Validate the model using KFold cross-validation.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    logging.info(f"Cross-validation R2 scores: {cv_results}")
    logging.info(f"Mean R2 score: {cv_results.mean()}")
    return cv_results


def plot_partial_dependence(model, X_train, features):
    """
    Plot partial dependence plots for the top features.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, features, ax=ax, grid_resolution=50
    )
    plt.suptitle("Partial Dependence Plots")
    plt.show()


def analyze_predictions(model, X_test, y_test):
    """
    Analyze the model's predictions to identify key factors influencing VM performance and resource allocation.
    Generate actionable recommendations to optimize resource usage and improve VM performance.
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Identify key factors (features) influencing VM performance
    feature_importances = model.feature_importances_
    important_features = X_test.columns[np.argsort(feature_importances)[::-1][:5]]

    # Generate actionable recommendations
    recommendations = []
    if "CPU Utilization (%)" in important_features:
        recommendations.append(
            "Consider optimizing CPU usage or upgrading CPUs for better performance."
        )
    if "Memory Utilization (%)" in important_features:
        recommendations.append(
            "Evaluate memory usage and consider adding more RAM if necessary."
        )
    if "Disk I/O Throughput (MB/s)" in important_features:
        recommendations.append(
            "Optimize disk I/O operations or upgrade storage hardware."
        )
    if "Network I/O Throughput (Mbps)" in important_features:
        recommendations.append("Enhance network bandwidth or optimize network usage.")
    if "System Load Average" in important_features:
        recommendations.append(
            "Monitor system load and distribute workloads more evenly."
        )

    logging.info("Key factors influencing VM performance: %s", important_features)
    logging.info("Recommendations: %s", recommendations)

    return important_features, recommendations


def visualize_recommendations(important_features, recommendations):
    """
    Visualize key factors and actionable recommendations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=important_features, y=[1] * len(important_features), palette="viridis", ax=ax
    )
    ax.set_title("Key Factors Influencing VM Performance")
    ax.set_xlabel("Important Features")
    ax.set_ylabel("")

    for i, rec in enumerate(recommendations):
        plt.text(i, 1.1, rec, ha="center", va="bottom", fontsize=10)

    plt.show()


def analyze_resource_utilization(df: pd.DataFrame):
    """
    Analyze resource utilization to identify potential bottlenecks.
    """
    resource_columns = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network I/O Throughput (Mbps)",
    ]

    # Identify potential bottlenecks by checking high utilization
    high_cpu = df["CPU Utilization (%)"] > 90
    high_memory = df["Memory Utilization (%)"] > 90
    high_network_io = df["Network I/O Throughput (Mbps)"] > 90

    bottlenecks = high_cpu | high_memory | high_network_io

    logging.info(f"Number of potential bottlenecks: {bottlenecks.sum()}")
    logging.info(f"High CPU Utilization: {high_cpu.sum()}")
    logging.info(f"High Memory Utilization: {high_memory.sum()}")
    logging.info(f"High Network I/O Throughput: {high_network_io.sum()}")

    return df[bottlenecks], resource_columns


def visualize_resource_utilization(df: pd.DataFrame, resource_columns: list):
    """
    Visualize resource utilization to identify bottlenecks.
    """
    plt.figure(figsize=(12, 8))

    for column in resource_columns:
        sns.histplot(df[column], kde=True, label=column)

    plt.title("Resource Utilization Analysis")
    plt.xlabel("Utilization")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def perform_clustering_analysis(df: pd.DataFrame, n_clusters: int = 3):
    """
    Perform clustering analysis on the dataset to group VMs based on resource usage patterns.
    """
    resource_columns = [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network I/O Throughput (Mbps)",
    ]

    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df[resource_columns])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(standardized_data)

    logging.info(f"Cluster centers: {kmeans.cluster_centers_}")

    return df, kmeans.cluster_centers_


def visualize_clusters_with_distribution(df: pd.DataFrame, cluster_centers):
    """
    Visualize overutilized, underutilized, and efficient VMs in one image along with the cluster distribution.
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Scatter plot of clusters
    sns.scatterplot(
        x="CPU Utilization (%)",
        y="Memory Utilization (%)",
        hue="Cluster",
        data=df,
        palette="viridis",
        alpha=0.6,
        edgecolor="w",
        ax=axs[0],
    )

    axs[0].scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        s=300,
        c="red",
        label="Cluster Centers",
        edgecolors="black",
    )

    axs[0].set_title(
        "Clustering Analysis of VMs: Overutilized, Underutilized, and Efficient VMs"
    )
    axs[0].set_xlabel("CPU Utilization (%)")
    axs[0].set_ylabel("Memory Utilization (%)")
    axs[0].legend(title="Cluster")
    axs[0].grid(True)

    # Bar plot of cluster distribution
    cluster_counts = df["Cluster"].value_counts().sort_index()
    sns.barplot(
        x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=axs[1]
    )
    axs[1].set_title("Number of Servers in Each Cluster")
    axs[1].set_xlabel("Cluster")
    axs[1].set_ylabel("Number of Servers")
    axs[1].grid(True)

    plt.show()

    logging.info(f"Cluster counts: {cluster_counts.to_dict()}")


def main():

    filepath = os.path.join("data", "system_metrics.csv")
    df = load_data(filepath)
    df = preprocess_data(df)

    # Display first few rows of the preprocessed data for verification
    logging.info(f"First few rows of the preprocessed data:\n{df.head()}")

    # Initial combined metrics plot
    plot_all_metrics_single_chart(df)

    # Perform EDA
    perform_eda(df)

    # Feature engineering
    df = feature_engineering(df)

    # Data Splitting
    target_column = "Resource_Utilization_Efficiency"
    X_train, X_test, y_train, y_test = split_data(df, target_column)

    # Handling Imbalanced Data
    X_train_balanced, y_train_balanced = handle_imbalanced_data_regression(
        X_train, y_train
    )

    # Train baseline model
    baseline_model = train_baseline_model(X_train_balanced, y_train_balanced)

    # Hyperparameter tuning
    best_model = hyperparameter_tuning(X_train_balanced, y_train_balanced)

    # Validate model
    validate_model(best_model, X_train_balanced, y_train_balanced)

    # Plot model performance
    plot_model_performance(best_model, X_test, y_test)

    # Plot residuals
    plot_residuals(y_test, best_model.predict(X_test))

    # Plot feature importances
    feature_names = X_train.columns
    plot_feature_importances(best_model, feature_names, top_n=10)

    # Insights and Recommendations
    important_features, recommendations = analyze_predictions(
        best_model, X_test, y_test
    )

    # Visualize recommendations
    visualize_recommendations(important_features, recommendations)

    # Partial Dependence Plots for the top features
    plot_partial_dependence(best_model, X_train, important_features)

    # Resource Utilization Analysis
    bottleneck_df, resource_columns = analyze_resource_utilization(df)

    # Clustering Analysis
    clustered_df, cluster_centers = perform_clustering_analysis(df)
    visualize_clusters_with_distribution(clustered_df, cluster_centers)


if __name__ == "__main__":
    main()
