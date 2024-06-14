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
        "Bubble Chart of Simulated Chaotic Server Metrics", fontsize=16, weight="bold"
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


def simulate_business_metrics(data):
    np.random.seed(42)
    data["Customer Satisfaction (CSAT)"] = np.random.normal(
        loc=90, scale=5, size=len(data)
    )
    data["Operational Costs ($)"] = np.random.normal(
        loc=1000, scale=200, size=len(data)
    )
    data["Service Uptime (%)"] = np.random.normal(loc=99, scale=0.5, size=len(data))
    data["Response Time (ms)"] = np.random.normal(loc=200, scale=50, size=len(data))
    return data


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
    cluster_business_summary = data.groupby("cluster")[BUSINESS_METRICS].mean()
    return cluster_business_summary


def generate_business_insights(cluster_business_summary):
    insights = []
    for cluster in cluster_business_summary.index:
        cluster_insights = f"\nCluster {cluster} Insights:"
        if cluster_business_summary.loc[cluster, "Response Time (ms)"] > 250:
            cluster_insights += "\n- High response time detected, consider optimizing server configurations to reduce response time."
        if cluster_business_summary.loc[cluster, "Customer Satisfaction (CSAT)"] < 85:
            cluster_insights += "\n- Low customer satisfaction detected, investigate potential causes related to server performance."
        if cluster_business_summary.loc[cluster, "Operational Costs ($)"] > 1200:
            cluster_insights += "\n- High operational costs detected, consider optimizing resource usage to reduce costs."
        if cluster_business_summary.loc[cluster, "Service Uptime (%)"] < 98:
            cluster_insights += "\n- Low service uptime detected, investigate and address potential causes to improve reliability."
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
        "PCA of Simulated Server Metrics with Clusters", fontsize=16, weight="bold"
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
        "t-SNE of Simulated Server Metrics with Clusters", fontsize=16, weight="bold"
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


def main():
    # Optional: Read data from a CSV file
    file_path = "netflix_weekly_metrics.csv"  # Update this path to your actual CSV file
    simulated_data = read_csv_file(file_path)

    # Step 1: Demonstrate the raw data that we have

    # Normalize the data
    normalizedData = normalize_data(simulated_data, FEATURES)

    # Plot the combined bubble chart
    plot_all_metrics_single_chart(normalizedData)

    # Display summary statistics
    # display_summary_statistics(simulated_data, FEATURES)

    # Plot summary statistics
    # plot_summary_statistics(simulated_data, FEATURES)

    # Plot pairwise relationships and distributions
    # plot_pairwise_relationships(simulated_data)

    # Plot the bubble chart
    plot_bubble_chart(simulated_data)

    # Step 2: Clean and Scale Data
    simulated_data, X_scaled_cleaned = clean_and_scale_data(simulated_data, FEATURES)

    # Step 3: Apply K-Means Clustering
    clusters_cleaned, kmeans = apply_kmeans_clustering(X_scaled_cleaned)
    simulated_data["cluster"] = clusters_cleaned

    # Step 4: Visualize Clusters
    visualize_clusters(simulated_data, FEATURES, clusters_cleaned)

    # Step 5: Apply PCA and t-SNE for Visualization
    pca_df, pca = apply_pca(X_scaled_cleaned, clusters_cleaned)
    pca_loadings = get_pca_loadings(pca, FEATURES)
    print("PCA Loadings:")
    print(pca_loadings)
    plot_pca_loadings(pca_loadings)

    # tsne_df = apply_tsne(X_scaled_cleaned, clusters_cleaned)

    # Step 6: Plot Distributions of Metrics within Clusters
    # plot_distributions(simulated_data, FEATURES)

    # Step 7: Analyze Distributions and Variances
    analysis_results = analyze_distributions(simulated_data, FEATURES)

    # Step 8: Generate Optimization Recommendations
    optimization_recommendations = generate_optimization_recommendations(
        analysis_results
    )
    for rec in optimization_recommendations:
        print(rec)

    # Step 9: Simulate Business Metrics
    simulated_data = simulate_business_metrics(simulated_data)

    # Step 10: Calculate Cluster Business Summary
    cluster_business_summary = calculate_cluster_business_summary(simulated_data)
    print("Cluster Business Summary:")
    print(cluster_business_summary)

    # Step 11: Generate Business Insights
    business_insights = generate_business_insights(cluster_business_summary)
    for insight in business_insights:
        print(insight)


if __name__ == "__main__":
    main()
