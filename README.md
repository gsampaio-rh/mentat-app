
# Mentat Server Metrics Analysis

## Introduction

This project aims to simulate and analyze server metrics to provide actionable business insights. The analysis includes clustering server performance data, correlating it with business metrics, and generating recommendations for optimization.

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
4. **Clustering:** Applies K-Means clustering to group server performance data.
5. **Profile Generation:** Creates profiles for each cluster based on average metrics.
6. **Insights Generation:** Provides optimization recommendations and business insights.
7. **Visualization:** Plots key metrics, temporal trends, and cluster insights.

## Results

The analysis provides:

- A correlation matrix between server performance and business metrics.
- Temporal trends of key business metrics.
- Profiles for each cluster.
- Best and worst performing clusters.
- Optimization recommendations and business insights.

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
