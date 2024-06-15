import pandas as pd
import logging


class Recommendations:
    @staticmethod
    def generate_optimization_recommendations(cluster_profiles):
        """
        Generate optimization recommendations based on cluster profiles.

        Args:
        - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles with summary statistics.

        Returns:
        - list: List of recommendations.
        """
        recommendations = []

        try:
            for idx, profile in cluster_profiles.iterrows():
                cluster_id = profile["Cluster"]
                recommendation = f"Cluster {cluster_id} Recommendations:\n"

                if profile.get("Mean CPU Utilization (%)", 0) > 80:
                    recommendation += "- CPU utilization is high. Consider load balancing or upgrading CPU capacity.\n"

                if profile.get("Mean Memory Utilization (%)", 0) > 80:
                    recommendation += "- Memory utilization is high. Consider optimizing memory usage or upgrading memory capacity.\n"

                if profile.get("Mean Network I/O Throughput (Mbps)", 0) > 1000:
                    recommendation += "- Network I/O throughput is high. Ensure network bandwidth is sufficient.\n"

                if profile.get("Mean Disk I/O Throughput (MB/s)", 0) > 500:
                    recommendation += "- Disk I/O throughput is high. Consider using faster storage solutions.\n"

                if profile.get("Mean Operational Costs ($)", 0) > 10000:
                    recommendation += "- Operational costs are high. Look into cost-saving measures.\n"

                if profile.get("Mean Customer Satisfaction (CSAT)", 0) < 70:
                    recommendation += "- Customer satisfaction is low. Investigate and address customer issues.\n"

                if profile.get("Mean Service Uptime (%)", 0) < 99:
                    recommendation += "- Service uptime is below the target. Improve system reliability.\n"

                if profile.get("Mean Response Time (ms)", 0) > 200:
                    recommendation += "- Response time is high. Optimize performance to reduce latency.\n"

                recommendations.append(recommendation.strip())
            logging.info("Optimization recommendations generated successfully.")
        except Exception as e:
            logging.error(f"An error occurred while generating recommendations: {e}")

        return recommendations

    @staticmethod
    def generate_business_insights(cluster_profiles):
        """
        Generate actionable business insights based on cluster profiles.

        Args:
        - cluster_profiles (pd.DataFrame): DataFrame containing the cluster profiles.

        Returns:
        - list: List of business insights.
        """
        insights = []
        try:
            for cluster in cluster_profiles.index:
                profile = cluster_profiles.loc[cluster]
                insights.append(f"Cluster {cluster} Insights:")
                if profile["Customer Satisfaction (CSAT)"] > 80:
                    insights.append(
                        " - High customer satisfaction. Continue current practices."
                    )
                elif profile["Customer Satisfaction (CSAT)"] < 50:
                    insights.append(
                        " - Low customer satisfaction. Investigate and address issues."
                    )

                if profile["Operational Costs ($)"] > 10000:
                    insights.append(
                        " - High operational costs. Look for optimization opportunities."
                    )
                elif profile["Operational Costs ($)"] < 5000:
                    insights.append(
                        " - Low operational costs. Evaluate if cost savings are affecting performance."
                    )

                if profile["Service Uptime (%)"] < 95:
                    insights.append(
                        " - Low service uptime. Improve reliability and reduce downtimes."
                    )

                insights.append("")  # Add a newline for better readability

            logging.info("Business insights generated successfully.")
        except Exception as e:
            logging.error(f"An error occurred while generating business insights: {e}")

        return insights
