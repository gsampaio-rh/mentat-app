import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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


def simulate_realistic_metrics(num_samples, config_name):
    np.random.seed(42)
    data = {
        "Server Configuration": [config_name] * num_samples,
        "CPU Utilization (%)": np.random.normal(loc=65, scale=15, size=num_samples),
        "Memory Utilization (%)": np.random.normal(loc=60, scale=10, size=num_samples),
        "Network I/O Throughput (Mbps)": np.random.normal(
            loc=40000, scale=10000, size=num_samples
        ),
        "Disk I/O Throughput (MB/s)": np.random.normal(
            loc=200, scale=50, size=num_samples
        ),
    }
    return pd.DataFrame(data)


def generate_simulated_data():
    num_samples = 1000
    configurations = ["Server A", "Server B", "Server C"]
    simulated_data = pd.concat(
        [simulate_realistic_metrics(num_samples, config) for config in configurations],
        ignore_index=True,
    )
    return simulated_data


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


def visualize_clusters(data, features, clusters):
    plt.figure(figsize=(12, 8))
    plt.scatter(
        data[features[0]],
        data[features[1]],
        s=data[features[2]] / 100,
        c=data[features[3]],
        alpha=0.5,
        cmap="viridis",
    )
    plt.colorbar(label=features[3])
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Bubble Chart of Simulated Realistic Server Metrics")
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.scatter(
        data[features[0]],
        data[features[1]],
        s=data[features[2]] / 100,
        c=clusters,
        alpha=0.5,
        cmap="viridis",
    )
    plt.colorbar(label="Cluster")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Bubble Chart of Simulated Realistic Server Metrics with Clusters")
    plt.show()


def apply_pca(scaled_data, clusters):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="viridis",
        data=pca_df,
        s=100,
        alpha=0.7,
    )
    plt.title("PCA of Simulated Server Metrics with Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()
    return pca_df, pca


def apply_tsne(scaled_data, clusters):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(scaled_data)
    tsne_df = pd.DataFrame(data=X_tsne, columns=["TSNE1", "TSNE2"])
    tsne_df["Cluster"] = clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="TSNE1",
        y="TSNE2",
        hue="Cluster",
        palette="viridis",
        data=tsne_df,
        s=100,
        alpha=0.7,
    )
    plt.title("t-SNE of Simulated Server Metrics with Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster")
    plt.show()
    return tsne_df


def get_pca_loadings(pca, features):
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=features)
    return loadings


def main():
    # Step 1: Generate Simulated Data
    simulated_data = generate_simulated_data()

    # Step 2: Clean and Scale Data
    simulated_data, X_scaled_cleaned = clean_and_scale_data(simulated_data, FEATURES)

    # Step 3: Apply K-Means Clustering
    clusters_cleaned, kmeans = apply_kmeans_clustering(X_scaled_cleaned)
    simulated_data["cluster"] = clusters_cleaned

    # Step 4: Visualize Clusters
    visualize_clusters(simulated_data, FEATURES, clusters_cleaned)

    # Step 5: Apply PCA and t-SNE for Visualization
    pca_df, pca = apply_pca(X_scaled_cleaned, clusters_cleaned)
    # t-SNE visualization might take too long, so it's commented out
    # tsne_df = apply_tsne(X_scaled_cleaned, clusters_cleaned)

    # Step 6: Plot Distributions of Metrics within Clusters
    plot_distributions(simulated_data, FEATURES)

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

    # Step 12: Display PCA Loadings
    loadings = get_pca_loadings(pca, FEATURES)
    print("PCA Loadings:")
    print(loadings)


if __name__ == "__main__":
    main()
