import pandas as pd
import matplotlib.pyplot as plt
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

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_metrics)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker="o", linestyle="-", color="skyblue")
plt.title("Determining the Optimal Number of Groups", fontsize=16)
plt.xlabel("Number of Clusters", fontsize=12)
plt.ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

# Highlight the elbow point
elbow_point = 4  # Assuming the elbow is at 4 clusters based on the graph
plt.plot(elbow_point, wcss[elbow_point - 1], marker="o", markersize=10, color="red")
plt.annotate(
    "Optimal number of groups identified here",
    xy=(elbow_point, wcss[elbow_point - 1]),
    xytext=(elbow_point + 1, wcss[elbow_point - 1] + 500),
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=12,
    color="red",
)

# Add a brief explanation above the chart
plt.figtext(
    0.5,
    0.9,
    "Using the Elbow Method to determine the optimal number of groups for clustering.",
    wrap=True,
    horizontalalignment="center",
    fontsize=14,
)

plt.show()

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_metrics)
combined_metrics["Cluster"] = kmeans.labels_

# System efficiency metrics
underutilized_vms = combined_metrics[combined_metrics["Cluster"] == 2].shape[
    0
]  # Assuming cluster 2 is underutilized
overutilized_vms = combined_metrics[combined_metrics["Cluster"] == 3].shape[
    0
]  # Assuming cluster 3 is overutilized
optimal_vms = combined_metrics[combined_metrics["Cluster"] == 1].shape[
    0
]  # Assuming cluster 1 is optimal

# Create a summary DataFrame
efficiency_summary = pd.DataFrame(
    {
        "Metric": ["Underutilized VMs", "Overutilized VMs", "Optimal VMs"],
        "Count": [underutilized_vms, overutilized_vms, optimal_vms],
    }
)

# Plot the summary of clustering results
plt.figure(figsize=(10, 6))
plt.bar(
    efficiency_summary["Metric"],
    efficiency_summary["Count"],
    color=["red", "orange", "green"],
)
plt.title("Clustering Analysis Results", fontsize=16)
plt.xlabel("VM Classification", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Display the summary DataFrame
print(efficiency_summary)

# Calculate cost savings from optimal resource allocation
cost_per_vm = 100  # Example cost per VM
total_vms = combined_metrics.shape[0]
current_cost = total_vms * cost_per_vm
optimized_cost = optimal_vms * cost_per_vm
cost_savings = current_cost - optimized_cost

# Plot cost savings
plt.figure(figsize=(10, 6))
categories = ["Current Cost", "Optimized Cost"]
values = [current_cost, optimized_cost]
plt.bar(categories, values, color=["skyblue", "lightgreen"])
plt.title("Cost Savings from Optimal Resource Allocation", fontsize=16)
plt.xlabel("Scenario", fontsize=12)
plt.ylabel("Cost ($)", fontsize=12)
plt.show()

# Plot predictive maintenance
plt.figure(figsize=(10, 6))
time = range(1, 11)  # Example time points
current_usage = combined_metrics["kubevirt_vmi_cpu_usage_seconds_total"][
    :10
]  # Example current usage
forecasted_usage = current_usage * 1.1  # Example forecast (10% increase)
plt.plot(time, current_usage, label="Current Usage", marker="o")
plt.plot(time, forecasted_usage, label="Forecasted Usage", linestyle="--", marker="o")
plt.title("Predictive Maintenance: Current vs. Forecasted Usage", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("CPU Usage (seconds)", fontsize=12)
plt.legend()
plt.show()
