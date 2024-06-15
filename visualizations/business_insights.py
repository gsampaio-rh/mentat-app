import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_business_insights_with_arrows(cluster_business_summary, business_insights):
    """
    Plot business insights with annotations.

    Args:
    - cluster_business_summary (pd.DataFrame): DataFrame containing the cluster business summary.
    - business_insights (list): List of business insights.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Cluster Business Summary Insights", fontsize=16)

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
        save_plot(fig, "business_insights_with_arrows.png")
        plt.show()
        logging.info("Business insights plotted successfully with annotations.")
    except Exception as e:
        logging.error(f"An error occurred while plotting business insights: {e}")
