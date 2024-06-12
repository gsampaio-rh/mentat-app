import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the datasets
kubevirt_metrics = pd.read_csv("data/kubevirt_metrics.csv")
openshift_metrics = pd.read_csv("data/openshift_metrics.csv")
insights_metrics = pd.read_csv("data/insights_metrics.csv")
rhel_metrics = pd.read_csv("data/rhel_metrics.csv")

# Select only numeric columns
kubevirt_metrics_numeric = kubevirt_metrics.select_dtypes(include=["number"])
openshift_metrics_numeric = openshift_metrics.select_dtypes(include=["number"])
insights_metrics_numeric = insights_metrics.select_dtypes(include=["number"])
rhel_metrics_numeric = rhel_metrics.select_dtypes(include=["number"])

# Fill missing values with the mean of the respective columns
kubevirt_metrics_numeric.fillna(kubevirt_metrics_numeric.mean(), inplace=True)
openshift_metrics_numeric.fillna(openshift_metrics_numeric.mean(), inplace=True)
insights_metrics_numeric.fillna(insights_metrics_numeric.mean(), inplace=True)
rhel_metrics_numeric.fillna(rhel_metrics_numeric.mean(), inplace=True)

# Combine relevant metrics for clustering
combined_metrics = pd.concat(
    [
        kubevirt_metrics_numeric[
            ["kubevirt_vmi_cpu_usage_seconds_total", "kubevirt_vmi_memory_used_bytes"]
        ],
        openshift_metrics_numeric[["Resource Request vs. Usage (CPU, memory)"]],
        rhel_metrics_numeric[["CPU Utilization (%)", "Memory Utilization (%)"]],
    ],
    axis=1,
)

# Standardize the data
scaler = StandardScaler()
scaled_metrics = scaler.fit_transform(combined_metrics)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_metrics)
combined_metrics["Cluster"] = kmeans.labels_

# Calculate cost savings from optimal resource allocation
# (Assume we have cost data per VM, here simplified as cost_per_vm)
cost_per_vm = 100  # Example cost per VM
total_vms = combined_metrics.shape[0]
optimized_vms = combined_metrics[combined_metrics["Cluster"] == 1].shape[
    0
]  # Assuming cluster 1 is optimal

current_cost = total_vms * cost_per_vm
optimized_cost = optimized_vms * cost_per_vm
cost_savings = current_cost - optimized_cost

# System efficiency metrics
underutilized_vms = combined_metrics[combined_metrics["Cluster"] == 2].shape[
    0
]  # Assuming cluster 2 is underutilized
overutilized_vms = combined_metrics[combined_metrics["Cluster"] == 3].shape[
    0
]  # Assuming cluster 3 is overutilized

# Plot cost savings
plt.figure(figsize=(10, 6))
categories = ["Current Cost", "Optimized Cost"]
values = [current_cost, optimized_cost]
plt.bar(categories, values, color=["skyblue", "lightgreen"])
plt.title("Cost Savings from Optimal Resource Allocation")
plt.xlabel("Scenario")
plt.ylabel("Cost ($)")
plt.show()

# Display system efficiency summary
efficiency_summary = pd.DataFrame(
    {
        "Metric": ["Underutilized VMs", "Overutilized VMs", "Optimal VMs"],
        "Count": [underutilized_vms, overutilized_vms, optimized_vms],
    }
)
print(efficiency_summary)

# Plot predictive maintenance
plt.figure(figsize=(10, 6))
time = range(1, 11)  # Example time points
current_usage = combined_metrics["kubevirt_vmi_cpu_usage_seconds_total"][
    :10
]  # Example current usage
forecasted_usage = current_usage * 1.1  # Example forecast (10% increase)
plt.plot(time, current_usage, label="Current Usage", marker="o")
plt.plot(time, forecasted_usage, label="Forecasted Usage", linestyle="--", marker="o")
plt.title("Predictive Maintenance: Current vs. Forecasted Usage")
plt.xlabel("Time")
plt.ylabel("CPU Usage (seconds)")
plt.legend()
plt.show()
