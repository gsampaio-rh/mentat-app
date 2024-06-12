import numpy as np
import pandas as pd

# Define the number of samples
num_samples = 1000

# Generate random data for each metric
rhel_metrics = {
    "CPU Utilization (%)": np.random.uniform(0, 100, num_samples),
    "Memory Utilization (%)": np.random.uniform(0, 100, num_samples),
    "Disk I/O Throughput (MB/s)": np.random.uniform(0, 100, num_samples),
    "Network I/O Throughput (Mbps)": np.random.uniform(0, 100, num_samples),
    "System Load Average": np.random.uniform(0, 10, num_samples),
    "Uptime (seconds)": np.random.uniform(0, 1000, num_samples),
    "Disk Space Utilization (%)": np.random.uniform(0, 100, num_samples),
    "Swap Utilization (%)": np.random.uniform(0, 100, num_samples),
    "Process Count": np.random.uniform(0, 100, num_samples),
    "Context Switch Rate": np.random.uniform(0, 100, num_samples),
}

insights_metrics = {
    "Number of Detected Vulnerabilities": np.random.uniform(0, 10, num_samples),
    "Anomaly Detection Alerts": np.random.uniform(0, 10, num_samples),
    "Compliance Policy Violations": np.random.uniform(0, 10, num_samples),
    "Risk Score": np.random.uniform(0, 10, num_samples),
    "Vulnerability Severity Distribution": np.random.uniform(0, 10, num_samples),
    "Compliance Drift (%)": np.random.uniform(0, 100, num_samples),
    "Vulnerability Exposure": np.random.uniform(0, 10, num_samples),
    "Baseline Compliance (%)": np.random.uniform(0, 100, num_samples),
    "Configuration Drift (%)": np.random.uniform(0, 100, num_samples),
    "Unauthorized Access Attempts": np.random.uniform(0, 10, num_samples),
}

openshift_metrics = {
    "Container Restart Rate": np.random.uniform(0, 10, num_samples),
    "Pod Lifecycle Duration (seconds)": np.random.uniform(0, 100, num_samples),
    "Resource Request vs. Usage (CPU, memory)": np.random.uniform(0, 10, num_samples),
    "Application Response Time (seconds)": np.random.uniform(0, 10, num_samples),
    "Deployment Success Rate (%)": np.random.uniform(0, 100, num_samples),
    "Cluster Node Health": np.random.choice(["healthy", "unhealthy"], num_samples),
    "Persistent Volume Usage (%)": np.random.uniform(0, 100, num_samples),
    "Network Traffic Patterns": np.random.choice(["incoming", "outgoing"], num_samples),
    "Horizontal Pod Autoscaler Efficiency (%)": np.random.uniform(0, 100, num_samples),
    "API Server Latency (seconds)": np.random.uniform(0, 10, num_samples),
}

kubevirt_metrics = {
    "kubevirt_vmi_cpu_usage_seconds_total": np.random.uniform(0, 100, num_samples),
    "kubevirt_vmi_memory_used_bytes": np.random.uniform(0, 100, num_samples),
    "kubevirt_vmi_network_receive_bytes_total": np.random.uniform(0, 100, num_samples),
    "kubevirt_vmi_network_transmit_bytes_total": np.random.uniform(0, 100, num_samples),
    "kubevirt_vmi_storage_read_traffic_bytes_total": np.random.uniform(
        0, 100, num_samples
    ),
    "kubevirt_vmi_storage_write_traffic_bytes_total": np.random.uniform(
        0, 100, num_samples
    ),
    "kubevirt_vmi_phase_count": np.random.uniform(0, 10, num_samples),
    "kubevirt_vmi_migration_data_processed_bytes": np.random.uniform(
        0, 100, num_samples
    ),
    "kubevirt_vmi_migration_data_remaining_bytes": np.random.uniform(
        0, 100, num_samples
    ),
    "kubevirt_vmi_filesystem_capacity_bytes": np.random.uniform(0, 100, num_samples),
}

# Create DataFrames for each category
rhel_df = pd.DataFrame(rhel_metrics)
insights_df = pd.DataFrame(insights_metrics)
openshift_df = pd.DataFrame(openshift_metrics)
kubevirt_df = pd.DataFrame(kubevirt_metrics)

# Save the DataFrames to CSV files
rhel_df.to_csv("rhel_metrics.csv", index=False)
insights_df.to_csv("insights_metrics.csv", index=False)
openshift_df.to_csv("openshift_metrics.csv", index=False)
kubevirt_df.to_csv("kubevirt_metrics.csv", index=False)
