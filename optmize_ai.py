import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib.colors import ListedColormap


def load_and_rename_dataset(file_path, rename_dict):
    """Load a dataset and rename its columns based on the provided dictionary."""
    df = pd.read_csv(file_path)
    return df.rename(columns=rename_dict)


def preprocess_data():
    """Load, rename, and combine all datasets."""
    # Define column rename mappings
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

    # Load datasets
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

    # Combine datasets
    combined_df = pd.concat(
        [kubevirt_metrics, openshift_metrics, insights_metrics, rhel_metrics], axis=1
    )

    return combined_df


def encode_non_numeric_columns(df):
    """Encode non-numeric columns in the dataframe."""
    label_encoder = LabelEncoder()
    df["Cluster Node Health"] = label_encoder.fit_transform(df["Cluster Node Health"])
    df["Network Traffic Patterns"] = label_encoder.fit_transform(
        df["Network Traffic Patterns"]
    )
    return df


def plot_correlation_matrix(df):
    """Plot the correlation matrix with highlighted significant correlations."""
    correlation_matrix = df.corr()
    mask = np.eye(correlation_matrix.shape[0], dtype=bool)
    gray_cmap = ListedColormap(["gray"])
    main_cmap = sns.diverging_palette(220, 20, as_cmap=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap=main_cmap,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Correlation Coefficient"},
        vmin=-0.1,
        vmax=0.1,
        center=0,
        annot_kws={"size": 6},
        mask=mask,
    )
    sns.heatmap(
        correlation_matrix,
        mask=~mask,
        cmap=gray_cmap,
        cbar=False,
        annot=False,
        linewidths=0.5,
    )

    threshold = 0.1
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if i != j and abs(correlation_matrix.iloc[i, j]) >= threshold:
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    f"{correlation_matrix.iloc[i, j]:.2f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="black",
                    fontsize=8,
                    weight="bold",
                )

    plt.title("Correlation Matrix of Metrics", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_distributions(df):
    """Plot the distribution of key metrics."""
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

    plt.figure(figsize=(20, 15))
    for i, metric in enumerate(key_metrics, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[metric], bins=30, kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def normalize_data(df):
    """Normalize the data using StandardScaler."""
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return normalized_df


def train_and_evaluate_model(df):
    """Train and evaluate a Random Forest Regressor model."""
    X = df.drop("CPU Usage (seconds)", axis=1)
    y = df["CPU Usage (seconds)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")


def main():
    combined_df = preprocess_data()
    combined_df = encode_non_numeric_columns(combined_df)
    plot_correlation_matrix(combined_df)
    plot_distributions(combined_df)
    normalized_df = normalize_data(combined_df)
    train_and_evaluate_model(normalized_df)


if __name__ == "__main__":
    main()
