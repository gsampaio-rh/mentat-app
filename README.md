# Mentat Server Metrics Analysis

## Introduction

This project aims to simulate and analyze server metrics to provide actionable business insights. The analysis includes clustering server performance data based on server configurations, correlating it with business metrics, and generating recommendations for optimization.

## Business Objectives

The main objectives of this project are:

1. To identify optimization opportunities within server operations.
2. To understand the correlation between server performance and business metrics.
3. To generate actionable insights and recommendations for improving operational efficiency and customer satisfaction.
4. To explain and demonstrate this analysis process in a way that is understandable for non-data science people.

## Problem Definition

The specific problem this project aims to solve is to:

- Analyze server performance data to identify patterns and correlations.
- Group servers with similar performance characteristics using clustering techniques.
- Provide insights and recommendations based on the analysis to optimize server operations and improve key business metrics such as customer satisfaction and operational costs.

## Key Metrics

The key metrics to be analyzed in this project include:

1. **Operational Metrics:**
   - CPU Utilization (%)
   - Memory Utilization (%)
   - Network I/O Throughput (Mbps)
   - Disk I/O Throughput (MB/s)

2. **Business Metrics:**
   - Customer Satisfaction (CSAT)
   - Operational Costs ($)
   - Service Uptime (%)
   - Response Time (ms)

## Data Simulation

The project simulates realistic server metrics data to perform the analysis. The data includes both operational and business metrics for a comprehensive understanding of server performance and its impact on business outcomes.

## Analysis Workflow

1. **Data Reading:** Reads operational and business metrics from CSV files.
2. **Data Merging:** Merges operational and business data for comprehensive analysis.
3. **Correlation Analysis:** Generates a correlation matrix between server performance and business metrics.
4. **Clustering:** Applies K-Means clustering to group server performance data based on server configurations.
5. **Profile Generation:** Creates profiles for each cluster based on average metrics and server configurations.
6. **Insights Generation:** Provides optimization recommendations and business insights.
7. **Visualization:** Plots key metrics, temporal trends, and cluster insights.

## Clusters Configuration

The server performance data is grouped into clusters based on different configurations:

- **EC2 API Servers (m5.large)**
- **EC2 Database Servers (r5.large)**
- **EC2 Streaming Servers (c5.large)**
- **EC2 Recommendation System Servers (p3.2xlarge)**

Each cluster represents a specific type of server configuration, which helps in understanding the performance characteristics and business impact of each type.

## Results

The analysis provides:

- A correlation matrix between server performance and business metrics.
- Temporal trends of key business metrics.
- Profiles for each cluster based on server configuration.
- Best and worst performing clusters.
- Optimization recommendations and business insights.

### Detailed Business Metrics and Insights

1. **Customer Satisfaction (CSAT)**
   - **Average CSAT Score:** Track the average customer satisfaction score over different time periods.
   - **CSAT vs. Server Performance:** Analyze how variations in CPU, memory, network I/O, and disk I/O impact customer satisfaction.
   - **CSAT Trends:** Identify trends and patterns in customer satisfaction over time.

2. **Operational Costs**
   - **Total Operational Costs:** Calculate the total operational costs associated with running the servers.
   - **Cost per Server:** Determine the operational cost for each server configuration.
   - **Cost Efficiency:** Identify the most cost-efficient server configurations based on their performance and business impact.
   - **Cost Reduction Opportunities:** Provide recommendations for reducing operational costs without compromising performance.

3. **Service Uptime**
   - **Overall Uptime Percentage:** Measure the overall uptime percentage across all server configurations.
   - **Uptime per Server:** Track the uptime percentage for each server configuration.
   - **Downtime Analysis:** Identify the root causes of downtime and suggest mitigation strategies.
   - **Impact of Uptime on Business Metrics:** Analyze how server uptime affects customer satisfaction and operational costs.

4. **Response Time**
   - **Average Response Time:** Calculate the average response time across all server configurations.
   - **Response Time per Server:** Track the response time for each server configuration.
   - **Response Time Trends:** Identify trends and patterns in server response times over different periods.
   - **Response Time Optimization:** Provide recommendations for optimizing response times based on server performance data.

5. **Server Performance Insights**
   - **High-Performance Clusters:** Identify clusters of servers that consistently perform well based on their configurations.
   - **Low-Performance Clusters:** Identify clusters of servers that require optimization based on their configurations.
   - **Performance Variability:** Measure the variability in server performance and its impact on business metrics.
   - **Performance Improvement Suggestions:** Provide actionable recommendations for improving server performance based on the clustering analysis.

6. **Business Impact Analysis**
   - **Correlation Analysis:** Show the correlation between operational metrics (CPU, memory, network I/O, disk I/O) and business metrics (CSAT, operational costs, uptime, response time).
   - **Key Drivers:** Identify the key drivers that have the most significant impact on business metrics.
   - **Scenario Analysis:** Conduct scenario analysis to predict the impact of changes in server performance on business outcomes.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/gsampaio-rh/mentat-app
    cd mentat-app
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Generate Data:** Run the script to generate simulated data.

    ```sh
    python generate_data.py
    ```

2. **Run Analysis:** Execute the main analysis script.

    ```sh
    python main.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
