import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Define constants for features and business metrics
FEATURES = [
    "CPU Utilization (%)",
    "Memory Utilization (%)",
    "Network I/O Throughput (Mbps)",
    "Disk I/O Throughput (MB/s)",
]
BUSINESS_METRICS = [
    "Customer Satisfaction (CSAT)",
    "Operational Costs ($)",
    "Service Uptime (%)",
    "Response Time (ms)",
]


def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.

    Args:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully read the file: {file_path}")
        return data
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return None


# Function to plot the summary statistics
def plot_summary_statistics(data, features):
    summary = data[features].describe().T
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap for the summary statistics
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    plt.title("Summary Statistics of Server Metrics", fontsize=16, weight="bold")

    plt.show()


# Function to normalize the data
def normalize_data(df, features):
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[features] = scaler.fit_transform(df_normalized[features])
    return df_normalized


# Function to plot all key metrics in a single chart
def plot_all_metrics_single_chart(df: pd.DataFrame):
    """
    Plot all key metrics in a single chart with different colors and markers.
    """
    plt.figure(figsize=(14, 8))

    # Define colors and markers for each metric
    colors = ["red", "green", "blue", "purple"]
    markers = ["o", "s", "D", "X"]

    # Plot each metric with a different color and marker
    for i, metric in enumerate(FEATURES):
        plt.scatter(
            df.index, df[metric], label=metric, color=colors[i], marker=markers[i], s=10
        )

    plt.title("All Key Metrics (Normalized)", fontsize=16, weight="bold")
    plt.xlabel("Index", fontsize=14)
    plt.ylabel("Normalized Value", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_bubble_chart(data):
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        data["CPU Utilization (%)"],
        data["Memory Utilization (%)"],
        s=(
            data["Network I/O Throughput (Mbps)"] / 500
        ),  # Adjusted for better visualization
        c=data["Disk I/O Throughput (MB/s)"],
        alpha=0.6,
        cmap="viridis",
        edgecolor="w",
        linewidth=0.5,
    )
    plt.colorbar(scatter, label="Disk I/O Throughput (MB/s)")
    plt.xlabel("CPU Utilization (%)", fontsize=14)
    plt.ylabel("Memory Utilization (%)", fontsize=14)
    plt.title(
        "Bubble Chart of Chaotic Server Metrics", fontsize=16, weight="bold"
    )
    plt.xlim(20, 120)  # Adjusted x-axis limits for better spread
    plt.ylim(10, 110)  # Adjusted y-axis limits for better spread
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def clean_and_scale_data(data, features):
    data[features] = data[features].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=features)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    return data, scaled_data


def apply_kmeans_clustering(scaled_data, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters, kmeans


def visualize_clusters(data, features, clusters):
    plt.figure(figsize=(21, 7))  # Adjusted size for three subplots

    # Define a custom color palette for clusters
    colors = sns.color_palette("Set2", len(np.unique(clusters)))

    sns.set(style="whitegrid")

    # First plot: unlabeled data
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(
        data[features[0]],
        data[features[1]],
        s=data[features[2]] / 500,
        c=data[features[3]],
        alpha=0.7,
        cmap="viridis",
        edgecolor="w",
        linewidth=0.5,
    )
    plt.colorbar(scatter, label=features[3])
    plt.xlabel(features[0], fontsize=14)
    plt.ylabel(features[1], fontsize=14)
    plt.title("Unlabeled Data", fontsize=16, weight="bold")
    plt.grid(True)
    plt.legend(
        ["Size: Network I/O Throughput", "Color: Disk I/O Throughput"],
        loc="upper right",
    )

    # Second plot: labeled clusters
    plt.subplot(1, 3, 2)
    for i, cluster in enumerate(np.unique(clusters)):
        cluster_data = data[clusters == cluster]
        plt.scatter(
            cluster_data[features[0]],
            cluster_data[features[1]],
            c=[colors[i]],
            alpha=0.7,
            label=f"Cluster {cluster}",
            edgecolors="w",  # Add a white edge to the points
            linewidth=0.5,
        )

    plt.xlabel(features[0], fontsize=14)  # Ensure the same x-axis label
    plt.ylabel(features[1], fontsize=14)  # Ensure the same y-axis label
    plt.title("Labeled Clusters", fontsize=16, weight="bold")
    plt.legend()
    plt.grid(True)

    # Third plot: cluster boundaries
    plt.subplot(1, 3, 3)
    for i, cluster in enumerate(np.unique(clusters)):
        cluster_data = data[clusters == cluster]
        plt.scatter(
            cluster_data[features[0]],
            cluster_data[features[1]],
            c=[colors[i]],
            alpha=0.7,
            label=f"Cluster {cluster}",
            edgecolors="w",  # Add a white edge to the points
            linewidth=0.5,
        )

        if len(cluster_data) > 2:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_data[features[:2]])
            hull_points = np.append(hull.vertices, hull.vertices[0])
            plt.plot(
                cluster_data[features[0]].values[hull_points],
                cluster_data[features[1]].values[hull_points],
                c=colors[i],
                alpha=0.7,
                linewidth=2,
            )
            # Fill the convex hull with a translucent color
            plt.fill(
                cluster_data[features[0]].values[hull_points],
                cluster_data[features[1]].values[hull_points],
                c=colors[i],
                alpha=0.2,
            )

    plt.xlabel(features[0], fontsize=14)  # Ensure the same x-axis label
    plt.ylabel(features[1], fontsize=14)  # Ensure the same y-axis label
    plt.title("Cluster Plot with Boundaries", fontsize=16, weight="bold")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_distributions(data, features, cluster_col="cluster"):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cluster_col, y=feature, data=data)
        plt.title(f"Distribution of {feature} by Cluster")
        plt.show()


def analyze_distributions(data, features, cluster_col="cluster"):
    analysis_results = {}
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        cluster_analysis = {}
        for feature in features:
            distribution = cluster_data[feature].describe()
            variance = cluster_data[feature].var()
            cluster_analysis[feature] = {
                "distribution": distribution,
                "variance": variance,
            }
        analysis_results[cluster] = cluster_analysis
    return analysis_results


def generate_optimization_recommendations(analysis_results):
    recommendations = []
    for cluster, metrics in analysis_results.items():
        rec = f"Cluster {cluster} Optimization Recommendations:\n"
        for feature, stats in metrics.items():
            if stats["variance"] > 50:
                rec += f"- {feature}: High variance detected, consider optimizing this metric.\n"
            if feature == "CPU Utilization (%)" and stats["distribution"]["mean"] > 80:
                rec += f"- {feature}: High average utilization, consider adding more CPU resources.\n"
            if (
                feature == "Memory Utilization (%)"
                and stats["distribution"]["mean"] > 70
            ):
                rec += f"- {feature}: High average utilization, consider adding more memory or optimizing memory usage.\n"
            if (
                feature == "Network I/O Throughput (Mbps)"
                and stats["distribution"]["mean"] > 30000
            ):
                rec += f"- {feature}: High network throughput, consider increasing network bandwidth.\n"
            if (
                feature == "Disk I/O Throughput (MB/s)"
                and stats["distribution"]["mean"] > 100
            ):
                rec += f"- {feature}: High disk throughput, consider using faster storage solutions.\n"
        recommendations.append(rec)
    return recommendations


# Function to plot pairwise relationships and distributions
def plot_pairwise_relationships(data):
    sns.set(style="whitegrid")
    pair_plot = sns.pairplot(
        data[FEATURES],
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20, "edgecolor": "k"},
    )
    pair_plot.fig.suptitle(
        "Pairwise Relationships and Distributions of Server Metrics",
        y=1.02,
        fontsize=16,
        weight="bold",
    )
    plt.show()


# Function to display summary statistics
def display_summary_statistics(data, features):
    summary = data[features].describe()
    print("Summary Statistics:")
    print(summary)


def calculate_cluster_business_summary(data):
    """
    Calculates the mean of business metrics for each cluster.

    Args:
    - data (pd.DataFrame): DataFrame containing the business metrics and cluster labels.

    Returns:
    - pd.DataFrame: DataFrame containing the mean of business metrics for each cluster.
    """
    cluster_business_summary = data.groupby("cluster")[BUSINESS_METRICS].mean()
    return cluster_business_summary


def generate_business_insights(cluster_business_summary):
    insights = []
    for cluster in cluster_business_summary.index:
        cluster_insights = f"\nCluster {cluster} Insights:"
        if cluster_business_summary.loc[cluster, "Response Time (ms)"] > 220:
            cluster_insights += "\n- High response time detected, consider optimizing server configurations to reduce response time."
        elif cluster_business_summary.loc[cluster, "Response Time (ms)"] > 190:
            cluster_insights += "\n- Response time could be improved. Analyze server performance and network latency."
        if cluster_business_summary.loc[cluster, "Customer Satisfaction (CSAT)"] < 90:
            cluster_insights += "\n- Customer satisfaction is lower than other clusters, investigate potential causes and gather customer feedback."
        if cluster_business_summary.loc[cluster, "Operational Costs ($)"] > 1100:
            cluster_insights += "\n- High operational costs detected, consider optimizing resource usage to reduce costs."
        elif cluster_business_summary.loc[cluster, "Operational Costs ($)"] > 1000:
            cluster_insights += "\n- Operational costs are relatively high. Investigate cost-saving measures without compromising service quality."
        if cluster_business_summary.loc[cluster, "Service Uptime (%)"] < 99:
            cluster_insights += "\n- Service uptime could be improved. Investigate and address potential causes to enhance reliability."
        insights.append(cluster_insights)
    return insights


# Function to apply PCA and plot the scatter plot
def apply_pca(scaled_data, clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters

    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")

    # Define a custom color palette for a clean look
    custom_palette = sns.color_palette("Set2", len(np.unique(clusters)))

    # Create the scatter plot with improved aesthetics
    scatter = sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=custom_palette,
        data=pca_df,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
    )

    # Enhance the legend for better readability
    legend = scatter.legend(
        title="Cluster",
        title_fontsize="13",
        fontsize="11",
        loc="upper right",
        frameon=True,
        shadow=True,
        borderpad=1,
    )

    plt.title(
        "PCA of Server Metrics with Clusters", fontsize=16, weight="bold"
    )
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)

    # Customize ticks and grid for a cleaner look
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)

    plt.show()
    return pca_df, pca


# Function to get PCA loadings
def get_pca_loadings(pca, features):
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
    return loadings


# Function to plot PCA loadings
def plot_pca_loadings(loadings):
    loadings.plot(kind="bar", figsize=(14, 8))
    plt.title(
        "PCA Loadings for Principal Components 1 and 2", fontsize=16, weight="bold"
    )
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Contribution to Principal Component", fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def apply_tsne(scaled_data, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=X_tsne, columns=["TSNE1", "TSNE2"])
    tsne_df["Cluster"] = clusters
    plt.figure(figsize=(14, 10))
    sns.set(style="whitegrid")

    # Define a custom color palette for a clean look
    custom_palette = sns.color_palette("Set2", len(np.unique(clusters)))

    sns.scatterplot(
        x="TSNE1",
        y="TSNE2",
        hue="Cluster",
        palette=custom_palette,
        data=tsne_df,
        s=100,
        alpha=0.8,
        edgecolor="w",
        linewidth=0.5,
    )
    plt.title(
        "t-SNE of Server Metrics with Clusters", fontsize=16, weight="bold"
    )
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(
        title="Cluster",
        title_fontsize="13",
        fontsize="11",
        loc="upper right",
        frameon=True,
        shadow=True,
        borderpad=1,
    )
    plt.show()
    return tsne_df


def plot_business_insights_with_arrows(cluster_business_summary, business_insights):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Cluster Business Summary Insights", fontsize=16)

    # Plot Customer Satisfaction (CSAT)
    axes[0, 0].bar(
        cluster_business_summary.index,
        cluster_business_summary["Customer Satisfaction (CSAT)"],
        color="skyblue",
    )
    axes[0, 0].set_title("Customer Satisfaction (CSAT)")
    axes[0, 0].set_xlabel("Cluster")
    axes[0, 0].set_ylabel("CSAT")
    if cluster_business_summary["Customer Satisfaction (CSAT)"].min() < 90:
        min_index = cluster_business_summary["Customer Satisfaction (CSAT)"].idxmin()
        axes[0, 0].annotate(
            "Lower satisfaction here",
            xy=(
                min_index,
                cluster_business_summary["Customer Satisfaction (CSAT)"][min_index],
            ),
            xytext=(
                min_index,
                cluster_business_summary["Customer Satisfaction (CSAT)"][min_index] + 1,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Operational Costs ($)
    axes[0, 1].bar(
        cluster_business_summary.index,
        cluster_business_summary["Operational Costs ($)"],
        color="lightgreen",
    )
    axes[0, 1].set_title("Operational Costs ($)")
    axes[0, 1].set_xlabel("Cluster")
    axes[0, 1].set_ylabel("Costs ($)")
    if cluster_business_summary["Operational Costs ($)"].max() > 1100:
        max_index = cluster_business_summary["Operational Costs ($)"].idxmax()
        axes[0, 1].annotate(
            "High operational cost",
            xy=(
                max_index,
                cluster_business_summary["Operational Costs ($)"][max_index],
            ),
            xytext=(
                max_index,
                cluster_business_summary["Operational Costs ($)"][max_index] + 50,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Service Uptime (%)
    axes[1, 0].bar(
        cluster_business_summary.index,
        cluster_business_summary["Service Uptime (%)"],
        color="lightcoral",
    )
    axes[1, 0].set_title("Service Uptime (%)")
    axes[1, 0].set_xlabel("Cluster")
    axes[1, 0].set_ylabel("Uptime (%)")
    if cluster_business_summary["Service Uptime (%)"].min() < 99:
        min_index = cluster_business_summary["Service Uptime (%)"].idxmin()
        axes[1, 0].annotate(
            "Slightly lower uptime",
            xy=(min_index, cluster_business_summary["Service Uptime (%)"][min_index]),
            xytext=(
                min_index,
                cluster_business_summary["Service Uptime (%)"][min_index] + 0.3,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )

    # Plot Response Time (ms)
    axes[1, 1].bar(
        cluster_business_summary.index,
        cluster_business_summary["Response Time (ms)"],
        color="lightblue",
    )
    axes[1, 1].set_title("Response Time (ms)")
    axes[1, 1].set_xlabel("Cluster")
    axes[1, 1].set_ylabel("Response Time (ms)")
    if cluster_business_summary["Response Time (ms)"].max() > 220:
        max_index = cluster_business_summary["Response Time (ms)"].idxmax()
        axes[1, 1].annotate(
            "High response time",
            xy=(max_index, cluster_business_summary["Response Time (ms)"][max_index]),
            xytext=(
                max_index,
                cluster_business_summary["Response Time (ms)"][max_index] + 20,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05),
            fontsize=12,
            color="red",
        )
    elif cluster_business_summary["Response Time (ms)"].max() > 190:
        max_index = cluster_business_summary["Response Time (ms)"].idxmax()
        axes[1, 1].annotate(
            "Moderate response time",
            xy=(max_index, cluster_business_summary["Response Time (ms)"][max_index]),
            xytext=(
                max_index,
                cluster_business_summary["Response Time (ms)"][max_index] + 20,
            ),
            arrowprops=dict(facecolor="orange", shrink=0.05),
            fontsize=12,
            color="orange",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def generate_correlation_matrix(data, features):
    correlation_matrix = data[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Business Metrics and Resources", fontsize=16)
    plt.show()
    return correlation_matrix


def plot_temporal_trends(data, metrics):
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data.set_index("Timestamp", inplace=True)
    plt.figure(figsize=(14, 8))
    for metric in metrics:
        data[metric].resample("D").mean().plot(label=metric)
    plt.title("Temporal Trends of Business Metrics", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def generate_cluster_profiles(data, features, business_metrics):
    cluster_profiles = data.groupby("cluster")[features + business_metrics].mean()
    print("Cluster Profiles:")
    print(cluster_profiles)
    return cluster_profiles


def identify_best_worst_clusters(cluster_profiles, metric):
    best_cluster = cluster_profiles[metric].idxmax()
    worst_cluster = cluster_profiles[metric].idxmin()
    return best_cluster, worst_cluster


def plot_resource_utilization_efficiency(cluster_profiles):
    cluster_profiles[
        [
            "CPU Utilization (%)",
            "Memory Utilization (%)",
            "Network I/O Throughput (Mbps)",
            "Disk I/O Throughput (MB/s)",
        ]
    ].plot(kind="bar", figsize=(14, 8), alpha=0.75, colormap="viridis")
    plt.title("Resource Utilization Efficiency by Cluster", fontsize=16, weight="bold")
    plt.xlabel("Cluster", fontsize=14)
    plt.ylabel("Utilization", fontsize=14)
    plt.legend(title="Metrics")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_cost_benefit_analysis(cluster_profiles):
    plt.figure(figsize=(14, 8))

    # Scatter plot for Operational Costs vs. Customer Satisfaction
    plt.scatter(
        cluster_profiles["Operational Costs ($)"],
        cluster_profiles["Customer Satisfaction (CSAT)"],
        s=200,
        alpha=0.75,
        label="CSAT vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    # Scatter plot for Operational Costs vs. Service Uptime
    plt.scatter(
        cluster_profiles["Operational Costs ($)"],
        cluster_profiles["Service Uptime (%)"],
        s=200,
        alpha=0.75,
        label="Uptime vs Costs",
        edgecolor="w",
        linewidth=0.5,
    )

    plt.title("Cost-Benefit Analysis", fontsize=16, weight="bold")
    plt.xlabel("Operational Costs ($)", fontsize=14)
    plt.ylabel("Performance Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

# Bar chart for average performance metrics for each server configuration
def plot_server_config_metrics(server_config_summary):
    server_config_summary[FEATURES].plot(kind='bar', figsize=(14, 8), alpha=0.75, colormap='viridis')
    plt.title('Average Performance Metrics by Server Configuration', fontsize=16, weight='bold')
    plt.xlabel('Server Configuration', fontsize=14)
    plt.ylabel('Average Metrics', fontsize=14)
    plt.legend(title='Metrics')
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.show()

# Scatter plot for Operational Costs vs. Key Performance Metrics for each server configuration
def plot_cost_vs_performance(server_config_summary):
    plt.figure(figsize=(14, 8))

    # Scatter plot for Operational Costs vs. Customer Satisfaction
    plt.scatter(server_config_summary['Operational Costs ($)'], server_config_summary['Customer Satisfaction (CSAT)'], s=200, alpha=0.75, label='CSAT vs Costs', edgecolor='w', linewidth=0.5)
    
    # Scatter plot for Operational Costs vs. Service Uptime
    plt.scatter(server_config_summary['Operational Costs ($)'], server_config_summary['Service Uptime (%)'], s=200, alpha=0.75, label='Uptime vs Costs', edgecolor='w', linewidth=0.5)

    plt.title('Cost vs Performance Metrics by Server Configuration', fontsize=16, weight='bold')
    plt.xlabel('Operational Costs ($)', fontsize=14)
    plt.ylabel('Performance Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.show()

def main():
    # Optional: Read data from a CSV file
    operational_file_path = (
        "data/netflix_operational_metrics.csv"  # Update this path to your actual CSV file
    )
    operational_data = read_csv_file(operational_file_path)

    business_file_path = "data/netflix_business_metrics.csv"  # Update this path to your actual CSV file
    business_data = read_csv_file(business_file_path)

    # Ensure the columns are correctly read
    print("Operational Data Columns:", operational_data.columns.tolist())
    print("Business Data Columns:", business_data.columns.tolist())

    # Step 1: Demonstrate the raw data that we have
    # Normalize the data
    normalizedData = normalize_data(operational_data, FEATURES)

    # Plot the combined bubble chart
    plot_all_metrics_single_chart(normalizedData)

    # Display summary statistics
    # display_summary_statistics(operational_data, FEATURES)

    # Plot summary statistics
    # plot_summary_statistics(operational_data, FEATURES)

    # Plot pairwise relationships and distributions
    # plot_pairwise_relationships(operational_data)

    # Plot the bubble chart
    plot_bubble_chart(operational_data)

    # Step 2: Clean and Scale Data
    operational_data, X_scaled_cleaned = clean_and_scale_data(
        operational_data, FEATURES
    )

    # Step 3: Apply K-Means Clustering
    clusters_cleaned, kmeans = apply_kmeans_clustering(X_scaled_cleaned)
    operational_data["cluster"] = clusters_cleaned

    # Step 4: Visualize Clusters
    visualize_clusters(operational_data, FEATURES, clusters_cleaned)

    # Step 5: Apply PCA and t-SNE for Visualization
    pca_df, pca = apply_pca(X_scaled_cleaned, clusters_cleaned)
    pca_loadings = get_pca_loadings(pca, FEATURES)
    print("PCA Loadings:")
    print(pca_loadings)
    plot_pca_loadings(pca_loadings)

    # tsne_df = apply_tsne(X_scaled_cleaned, clusters_cleaned)

    # Step 6: Plot Distributions of Metrics within Clusters
    # plot_distributions(operational_data, FEATURES)

    # Step 7: Analyze Distributions and Variances
    analysis_results = analyze_distributions(operational_data, FEATURES)

    # Step 8: Generate Optimization Recommendations
    optimization_recommendations = generate_optimization_recommendations(
        analysis_results
    )
    for rec in optimization_recommendations:
        print(rec)

    # Ensure the cluster labels are also in the business data
    business_data['cluster'] = operational_data['cluster']

    # Step 9: Calculate Cluster Business Summary
    cluster_business_summary = calculate_cluster_business_summary(business_data)
    print("Cluster Business Summary:")
    print(cluster_business_summary)

    # Step 10: Generate Business Insights
    # Generate Correlation Matrix
    # Merge operational and business data for correlation analysis
    combined_data = pd.merge(
        operational_data,
        business_data,
        on=["Timestamp", "Server Configuration", "cluster"],
    )

    correlation_matrix = generate_correlation_matrix(
        combined_data, BUSINESS_METRICS + FEATURES
    )

    # Plot Temporal Trends
    plot_temporal_trends(business_data, BUSINESS_METRICS)

    # Generate Cluster Profiles
    cluster_profiles = generate_cluster_profiles(
        combined_data, FEATURES, BUSINESS_METRICS
    )

    plot_resource_utilization_efficiency(cluster_profiles)

    plot_cost_benefit_analysis(cluster_profiles)

    # Identify Best and Worst Performing Clusters
    best_cluster, worst_cluster = identify_best_worst_clusters(
        cluster_profiles, "Customer Satisfaction (CSAT)"
    )
    print(f"Best Performing Cluster: {best_cluster}")
    print(f"Worst Performing Cluster: {worst_cluster}")

    business_insights = generate_business_insights(cluster_business_summary)
    # print(business_insights)

    plot_business_insights_with_arrows(cluster_business_summary, business_insights)
    for insight in business_insights:
        print(insight)

    # Aggregate metrics by Server Configuration
    server_config_summary = combined_data.groupby("Server Configuration")[
        FEATURES + BUSINESS_METRICS
    ].mean()

    # Display the summary
    server_config_summary

    scaler = MinMaxScaler()
    normalized_server_config_summary = pd.DataFrame(
        scaler.fit_transform(server_config_summary),
        columns=server_config_summary.columns,
        index=server_config_summary.index,
    )

    plot_server_config_metrics(normalized_server_config_summary)
    plot_cost_vs_performance(server_config_summary)

if __name__ == "__main__":
    main()
