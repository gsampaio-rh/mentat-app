import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib.colors import ListedColormap


# Function to load and rename dataset columns
def load_and_rename_dataset(file_path, rename_dict):
    df = pd.read_csv(file_path)
    return df.rename(columns=rename_dict)


# Load the datasets
kubevirt_metrics = load_and_rename_dataset(
    "data/kubevirt_metrics.csv",
    {
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
    },
)

openshift_metrics = load_and_rename_dataset(
    "data/openshift_metrics.csv",
    {
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
    },
)

insights_metrics = load_and_rename_dataset(
    "data/insights_metrics.csv",
    {
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
    },
)

rhel_metrics = load_and_rename_dataset(
    "data/rhel_metrics.csv",
    {
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
    },
)

# Combine datasets on the index as a common basis
combined_df = pd.concat(
    [
        kubevirt_metrics,
        openshift_metrics,
        insights_metrics,
        rhel_metrics,
    ],
    axis=1,
)

# Inspect problematic columns
print(
    f'Unique values in Cluster Node Health: {combined_df["Cluster Node Health"].unique()}'
)
print(
    f'Unique values in Network Traffic Patterns: {combined_df["Network Traffic Patterns"].unique()}'
)

# Handle non-numeric values using Label Encoding
label_encoder = LabelEncoder()
combined_df["Cluster Node Health"] = label_encoder.fit_transform(
    combined_df["Cluster Node Health"]
)
combined_df["Network Traffic Patterns"] = label_encoder.fit_transform(
    combined_df["Network Traffic Patterns"]
)

# Recalculate the correlation matrix
correlation_matrix = combined_df.corr()

# Create a mask for the diagonal
mask = np.eye(correlation_matrix.shape[0], dtype=bool)

# Create a custom colormap with gray for the mask
gray_cmap = ListedColormap(["gray"])
main_cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Improved Correlation Matrix Visualization with highlighted significant correlations
plt.figure(figsize=(12, 10))  # Adjusted figure size for a smaller fit

# Define a more focused range for the color map
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
    annot_kws={"size": 6},  # Smaller font size for annotations
    mask=mask,  # Mask the diagonal
)

# Overlay the mask with gray
sns.heatmap(
    correlation_matrix,
    mask=~mask,
    cmap=gray_cmap,
    cbar=False,
    annot=False,
    linewidths=0.5,
)

# Highlight significant correlations excluding the diagonal
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

plt.title("Correlation Matrix of Metrics", fontsize=14)  # Smaller title font size
plt.xticks(rotation=90, fontsize=8)  # Smaller x-tick font size
plt.yticks(fontsize=8)  # Smaller y-tick font size
plt.tight_layout()  # Ensures the layout fits into the window
plt.show()

# Distribution plots
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
    sns.histplot(combined_df[metric], bins=30, kde=True)
    plt.title(f"Distribution of {metric}")
    plt.xlabel(metric)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Feature engineering: Normalize data
scaler = StandardScaler()
normalized_df = pd.DataFrame(
    scaler.fit_transform(combined_df), columns=combined_df.columns
)

# Splitting the data into training and testing sets
X = normalized_df.drop(
    "CPU Usage (seconds)", axis=1
)  # Assuming CPU Usage is the target
y = normalized_df["CPU Usage (seconds)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training: Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
