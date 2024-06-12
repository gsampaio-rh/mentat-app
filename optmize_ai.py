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

    plt.figure(figsize=(20, 15))
    for i, metric in enumerate(key_metrics, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[metric], bins=30, kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")

        insights = calculate_distribution_insights(df, metric)
        for insight in insights:
            plt.gca().annotate(
                insight,
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
                color="red",
            )

    plt.tight_layout()
    plt.show()


def normalize_data(df):
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return normalized_df


def train_and_evaluate_model(df):
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


def main():
    combined_df = preprocess_data()
    combined_df = encode_non_numeric_columns(combined_df)
    plot_original_correlation_matrix(combined_df)
    plot_filtered_correlation_matrix(combined_df)
    plot_distributions(combined_df)
    normalized_df = normalize_data(combined_df)
    train_and_evaluate_model(normalized_df)


if __name__ == "__main__":
    main()
