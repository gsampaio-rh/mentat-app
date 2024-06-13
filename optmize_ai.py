import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
import numpy as np

def load_and_rename_dataset(file_path, rename_dict):
    df = pd.read_csv(file_path)
    return df.rename(columns=rename_dict)


def preprocess_data():
    kubevirt_rename = {
        "kubevirt_vmi_cpu_usage_seconds_total": "CPU Usage (seconds)",
        "kubevirt_vmi_memory_used_bytes": "Memory Usage (bytes)",
        "kubevirt_vmi_network_receive_bytes_total": "Network Receive (bytes)",
        "kubevirt_vmi_network_transmit_bytes_total": "Network Transmit (bytes)",
        "kubevirt_vmi_storage_read_traffic_bytes_total": "Storage Read (bytes)",
        "kubevirt_vmi_storage_write_traffic_bytes_total": "Storage Write (bytes)",
        "kubevirt_vmi_phase_count": "Phase Count",
        "kubevirt_vmi_migration_data_processed_bytes": "Migration Data Processed (bytes)",
        "kubevirt_vmi_migration_data_remaining_bytes": "Migration Data Remaining (bytes)",
        "kubevirt_vmi_filesystem_capacity_bytes": "Filesystem Capacity (bytes)",
    }

    openshift_rename = {
        "CPU Utilization (%)": "CPU Utilization (%)",
        "Memory Utilization (%)": "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)": "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)": "Network I/O Throughput (Mbps)",
        "System Load Average": "System Load Average",
        "Uptime (seconds)": "Uptime (seconds)",
        "Disk Space Utilization (%)": "Disk Space Utilization (%)",
        "Swap Utilization (%)": "Swap Utilization (%)",
        "Process Count": "Process Count",
        "Context Switch Rate": "Context Switch Rate",
    }

    insights_rename = {
        "Number of Detected Vulnerabilities": "Detected Vulnerabilities",
        "Anomaly Detection Alerts": "Anomaly Detection Alerts",
        "Compliance Policy Violations": "Policy Violations",
        "Risk Score": "Risk Score",
        "Vulnerability Severity Distribution": "Vulnerability Severity",
        "Compliance Drift (%)": "Compliance Drift (%)",
        "Vulnerability Exposure": "Vulnerability Exposure",
        "Baseline Compliance (%)": "Baseline Compliance (%)",
        "Configuration Drift (%)": "Configuration Drift (%)",
        "Unauthorized Access Attempts": "Unauthorized Access Attempts",
    }

    rhel_rename = {
        "CPU Utilization (%)": "CPU Utilization (%)",
        "Memory Utilization (%)": "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)": "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)": "Network I/O Throughput (Mbps)",
        "System Load Average": "System Load Average",
        "Uptime (seconds)": "Uptime (seconds)",
        "Disk Space Utilization (%)": "Disk Space Utilization (%)",
        "Swap Utilization (%)": "Swap Utilization (%)",
        "Process Count": "Process Count",
        "Context Switch Rate": "Context Switch Rate",
    }

    kubevirt_metrics = load_and_rename_dataset(
        "data/kubevirt_metrics.csv", kubevirt_rename
    )
    openshift_metrics = load_and_rename_dataset(
        "data/openshift_metrics.csv", openshift_rename
    )
    insights_metrics = load_and_rename_dataset(
        "data/insights_metrics.csv", insights_rename
    )
    rhel_metrics = load_and_rename_dataset("data/rhel_metrics.csv", rhel_rename)

    combined_df = pd.concat(
        [kubevirt_metrics, openshift_metrics, insights_metrics, rhel_metrics], axis=1
    )
    return combined_df


def encode_non_numeric_columns(df):
    label_encoder = LabelEncoder()
    df["Cluster Node Health"] = label_encoder.fit_transform(df["Cluster Node Health"])
    df["Network Traffic Patterns"] = label_encoder.fit_transform(
        df["Network Traffic Patterns"]
    )
    return df


def normalize_data(df):
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return normalized_df


def plot_heatmap(correlation_matrix, mask, title, cmap, threshold):
    correlation_matrix *= 10
    plt.figure(figsize=(12, 10))
    significant_mask = (correlation_matrix.abs() >= threshold) & (mask == False)

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws={"size": 6},
        mask=~significant_mask,
    )
    sns.heatmap(
        correlation_matrix,
        mask=mask | significant_mask,
        cmap=ListedColormap(["gray"]),
        cbar=False,
        annot=False,
        linewidths=0.5,
    )

    plt.title(title, fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def log_filtered_correlations(correlation_matrix, threshold):
    significant_correlations = correlation_matrix[
        (correlation_matrix.abs() >= threshold) & (correlation_matrix.abs() < 1)
    ]
    significant_correlations = significant_correlations.stack().reset_index()
    significant_correlations.columns = ["Metric 1", "Metric 2", "Correlation"]
    significant_correlations = significant_correlations.sort_values(
        by="Correlation", ascending=False
    )
    print(significant_correlations.to_string(index=False))


def plot_original_correlation_matrix(df):
    correlation_matrix = df.corr()
    mask = np.eye(correlation_matrix.shape[0], dtype=bool)
    plot_heatmap(
        correlation_matrix,
        mask,
        "Original Correlation Matrix of Metrics",
        sns.diverging_palette(220, 20, as_cmap=True),
        threshold=-1,
    )


def plot_filtered_correlation_matrix(df, threshold=0.7):
    correlation_matrix = df.corr()
    mask = np.eye(correlation_matrix.shape[0], dtype=bool)
    plot_heatmap(
        correlation_matrix,
        mask,
        "Filtered Correlation Matrix of Metrics with Highlighted Significant Correlations",
        sns.diverging_palette(220, 20, as_cmap=True),
        threshold=threshold,
    )
    log_filtered_correlations(correlation_matrix, threshold)


def calculate_distribution_insights(df, metric):
    insights = []
    skewness = df[metric].skew()
    if skewness > 1:
        insights.append(f"{metric} is highly positively skewed.")
    elif skewness < -1:
        insights.append(f"{metric} is highly negatively skewed.")
    else:
        insights.append(f"{metric} is approximately symmetric.")

    # Check for unimodal or multimodal distribution using numpy
    data = df[metric].dropna().values
    density, bins = np.histogram(data, bins=30, density=True)
    peaks = (np.diff(np.sign(np.diff(density))) < 0).sum()
    if peaks == 1:
        insights.append(f"{metric} has a single peak (unimodal).")
    elif peaks > 1:
        insights.append(f"{metric} has multiple peaks (multimodal).")

    # Calculate range
    data_range = df[metric].max() - df[metric].min()
    insights.append(f"The range of {metric} is {data_range:.2f}.")

    # Identify outliers
    Q1 = df[metric].quantile(0.25)
    Q3 = df[metric].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[metric] < (Q1 - 1.5 * IQR)) | (df[metric] > (Q3 + 1.5 * IQR))).sum()
    if outliers > 0:
        insights.append(f"{metric} has {outliers} outliers.")

    # Calculate standard deviation
    std_dev = df[metric].std()
    insights.append(f"The standard deviation of {metric} is {std_dev:.2f}.")

    return insights


def plot_distributions(df):
    key_metrics = [
        "CPU Usage (seconds)",
        "Memory Usage (bytes)",
        "Network Receive (bytes)",
        "Network Transmit (bytes)",
        "Storage Read (bytes)",
        "Storage Write (bytes)",
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Disk I/O Throughput (MB/s)",
        "Network I/O Throughput (Mbps)",
    ]

    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(key_metrics, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[metric], bins=30, kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")

        insights = calculate_distribution_insights(df, metric)
        for j, insight in enumerate(insights):
            plt.gca().annotate(
                insight,
                xy=(0.5, 0.9 - j * 0.05),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="red",
            )

    plt.tight_layout()
    plt.show()


def visualize_predictions(df):
    X = df.drop("CPU Usage (seconds)", axis=1)
    y = df["CPU Usage (seconds)"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plotting Actual vs. Predicted CPU Usage
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted Values")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r",
        linewidth=2,
        label="Ideal Prediction Line",
    )
    plt.xlabel("Actual CPU Usage (seconds)")
    plt.ylabel("Predicted CPU Usage (seconds)")
    plt.title("Actual vs. Predicted CPU Usage")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculating and displaying the error metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return mae, rmse


def feature_engineering(df):
    # Create CPU utilization ratio
    df["CPU Utilization Ratio"] = df["CPU Usage (seconds)"] / df["CPU Utilization (%)"]

    # Create memory utilization ratio
    df["Memory Utilization Ratio"] = (
        df["Memory Usage (bytes)"] / df["Memory Utilization (%)"]
    )

    # Create network I/O ratio
    df["Network I/O Ratio"] = (
        df["Network Receive (bytes)"] + df["Network Transmit (bytes)"]
    ) / df["Network I/O Throughput (Mbps)"]

    # Create storage I/O ratio
    df["Storage I/O Ratio"] = (
        df["Storage Read (bytes)"] + df["Storage Write (bytes)"]
    ) / df["Disk I/O Throughput (MB/s)"]

    # Create system load per core (assuming 4 cores for this example)
    df["System Load Per Core"] = df["System Load Average"] / 4

    # Indicator for peak usage times (example: usage above 90%)
    df["Peak CPU Usage"] = (df["CPU Utilization (%)"] > 90).astype(int)
    df["Peak Memory Usage"] = (df["Memory Utilization (%)"] > 90).astype(int)
    df["Peak Network Usage"] = (df["Network I/O Throughput (Mbps)"] > 90).astype(int)
    df["Peak Storage Usage"] = (df["Disk I/O Throughput (MB/s)"] > 90).astype(int)

    return df


def train_bottleneck_model(df):
    # Define target variable (bottleneck indicator: 1 if any peak usage, else 0)
    df["Bottleneck"] = df[
        [
            "Peak CPU Usage",
            "Peak Memory Usage",
            "Peak Network Usage",
            "Peak Storage Usage",
        ]
    ].max(axis=1)

    # Ensure feature columns are consistent
    feature_cols = [
        "CPU Utilization Ratio",
        "Memory Utilization Ratio",
        "Network I/O Ratio",
        "Storage I/O Ratio",
        "System Load Per Core",
        "Detected Vulnerabilities",
        "Risk Score",
    ]
    X = df[feature_cols]
    y = df["Bottleneck"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict bottlenecks on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return model, feature_cols, X_test, y_test, y_pred


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


def plot_classification_results_with_clusters(X_test, y_test, y_pred, df):
    # Ensure "Number of Servers" column is present and has appropriate values
    if "Number of Servers" not in df.columns:
        print("Warning: 'Number of Servers' column not found. Adding placeholder values.")
        df["Number of Servers"] = np.random.randint(1, 100, size=len(df))
    else:
        # If the column exists, ensure it has reasonable values
        if df["Number of Servers"].isnull().any() or (df["Number of Servers"] <= 0).any():
            print("Warning: 'Number of Servers' column has invalid values. Replacing with placeholder values.")
            df["Number of Servers"] = np.random.randint(1, 100, size=len(df))

    # Fit a Gaussian Mixture Model to get ellipses representing clusters
    n_components = min(len(X_test), 3)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_test.iloc[:, :2])  # Fit only the first two features for visualization

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    sizes = df.loc[X_test.index, "Number of Servers"] * 10  # Scale for better visualization
    scatter = ax.scatter(
        X_test.iloc[:, 0],
        X_test.iloc[:, 1],
        c=y_pred,
        s=sizes,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="w",
        label=y_pred
    )
    plot_ellipses(gmm, ax)
    plt.xlabel("CPU Utilization Ratio")
    plt.ylabel("Memory Utilization Ratio")
    plt.title("Random Forest Classification Results with Server Counts and Clusters")

    # Add a legend
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend1 = ax.legend(handles, labels, loc="upper right", title="Number of Servers")
    ax.add_artist(legend1)

    handles, labels = scatter.legend_elements()
    legend2 = ax.legend(handles, ["Non-Bottleneck", "Bottleneck"], loc="upper left", title="Prediction")
    ax.add_artist(legend2)

    plt.show()



def visualize_bottlenecks(df, model):
    feature_cols = [
        "CPU Utilization Ratio",
        "Memory Utilization Ratio",
        "Network I/O Ratio",
        "Storage I/O Ratio",
        "System Load Per Core",
        "Detected Vulnerabilities",
        "Risk Score",
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

# Visualize feature importance
def plot_feature_importance(model, feature_cols):
    feature_importance = pd.Series(model.feature_importances_, index=feature_cols)
    feature_importance = feature_importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title("Feature Importance in Bottleneck Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


def main():
    combined_df = preprocess_data()
    combined_df = encode_non_numeric_columns(combined_df)

    # plot_original_correlation_matrix(combined_df)
    # plot_filtered_correlation_matrix(combined_df)
    # plot_distributions(combined_df)

    # Feature engineering
    combined_df = feature_engineering(combined_df)

    # Train the model and get predictions
    bottleneck_model, feature_cols, X_test, y_test, y_pred = train_bottleneck_model(
        combined_df
    )

    # Visualize the classification results with bubble sizes and ellipses
    plot_classification_results_with_clusters(X_test, y_test, y_pred, combined_df)

    # Visualize the

    # Visualize the predicted bottlenecks
    visualize_bottlenecks(combined_df, bottleneck_model)
    plot_feature_importance(bottleneck_model, feature_cols)
    # plot_bottlenecks_scatter(combined_df)

    normalized_df = normalize_data(combined_df)
    # train_and_evaluate_model(normalized_df)
    # visualize_predictions(
    #     normalized_df
    # # )
    # bottleneck_model = train_bottleneck_model(normalized_df)
    # visualize_bottlenecks(normalized_df, bottleneck_model)


if __name__ == "__main__":
    main()
