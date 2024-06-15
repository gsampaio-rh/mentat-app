import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull
from utils.utils import save_plot
import logging

def visualize_clusters(data, cluster_labels, features):
    """
    Visualize clusters using pair plots.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - cluster_labels (str): Column name for cluster labels.
    - features (list): List of features to include in the plots.
    """
    try:
        sns.pairplot(data[features + [cluster_labels]], hue=cluster_labels)
        plt.show()
        logging.info("Clusters visualized successfully.")
    except Exception as e:
        logging.error(f"An error occurred while visualizing clusters: {e}")

def visualize_all_labelled_non_labelled_clusters(data, features, clusters):
    """
    Visualize clusters using scatter plots.

    Args:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of features to plot.
    - clusters (array): Array of cluster labels.
    """
    try:
        fig = plt.figure(figsize=(21, 7))
        colors = sns.color_palette("Set2", len(np.unique(clusters)))
        sns.set(style="whitegrid")

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

        plt.subplot(1, 3, 2)
        for i, cluster in enumerate(np.unique(clusters)):
            cluster_data = data[clusters == cluster]
            plt.scatter(
                cluster_data[features[0]],
                cluster_data[features[1]],
                c=[colors[i]],
                alpha=0.7,
                label=f"Cluster {cluster}",
                edgecolors="w",
                linewidth=0.5,
            )

        plt.xlabel(features[0], fontsize=14)
        plt.ylabel(features[1], fontsize=14)
        plt.title("Labeled Clusters", fontsize=16, weight="bold")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        for i, cluster in enumerate(np.unique(clusters)):
            cluster_data = data[clusters == cluster]
            plt.scatter(
                cluster_data[features[0]],
                cluster_data[features[1]],
                c=[colors[i]],
                alpha=0.7,
                label=f"Cluster {cluster}",
                edgecolors="w",
                linewidth=0.5,
            )
            if len(cluster_data) > 2:
                hull = ConvexHull(cluster_data[features[:2]])
                hull_points = np.append(hull.vertices, hull.vertices[0])
                plt.plot(
                    cluster_data[features[0]].values[hull_points],
                    cluster_data[features[1]].values[hull_points],
                    c=colors[i],
                    alpha=0.7,
                    linewidth=2,
                )
                plt.fill(
                    cluster_data[features[0]].values[hull_points],
                    cluster_data[features[1]].values[hull_points],
                    c=colors[i],
                    alpha=0.2,
                )

        plt.xlabel(features[0], fontsize=14)
        plt.ylabel(features[1], fontsize=14)
        plt.title("Cluster Plot with Boundaries", fontsize=16, weight="bold")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        save_plot(fig, "visualize_clusters.png")
        plt.show()
        logging.info("All labeled and non-labeled clusters visualized successfully.")
    except Exception as e:
        logging.error(f"An error occurred while visualizing all clusters: {e}")
