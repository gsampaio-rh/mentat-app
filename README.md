# Mentat Server Metrics Analysis

This project simulates and analyzes server metrics to provide actionable business insights. The analysis includes clustering server performance data, correlating it with business metrics, and generating recommendations for optimization.

## Table of Contents

- [Mentat Server Metrics Analysis](#mentat-server-metrics-analysis)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data Simulation](#data-simulation)
  - [Analysis Workflow](#analysis-workflow)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

This project aims to generate realistic server metrics, perform clustering analysis, and derive business insights. It helps identify optimization opportunities and understand the correlation between server performance and business metrics.

## Data Simulation

The script simulates operational and business metrics for servers, including:

- **Operational Metrics**: CPU Utilization, Memory Utilization, Network I/O Throughput, Disk I/O Throughput.
- **Business Metrics**: Customer Satisfaction (CSAT), Operational Costs, Service Uptime, Response Time.

## Analysis Workflow

1. **Data Reading**: Reads operational and business metrics from CSV files.
2. **Data Merging**: Merges operational and business data for comprehensive analysis.
3. **Correlation Analysis**: Generates a correlation matrix between server performance and business metrics.
4. **Clustering**: Applies K-Means clustering to group server performance data.
5. **Profile Generation**: Creates profiles for each cluster based on average metrics.
6. **Insights Generation**: Provides optimization recommendations and business insights.
7. **Visualization**: Plots key metrics, temporal trends, and cluster insights.

## Features

- **Simulated Data**: Generates realistic server metrics data.
- **Clustering**: Identifies clusters of server performance.
- **Correlation Analysis**: Analyzes the relationship between operational metrics and business outcomes.
- **Optimization Recommendations**: Provides actionable insights for each cluster.
- **Visualization**: Visualizes data trends and insights.

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

1. **Generate Data**: Run the script to generate simulated data.

    ```sh
    python generate_data.py
    ```

2. **Run Analysis**: Execute the main analysis script.

    ```sh
    python main.py
    ```

## Results

The analysis provides:

- Correlation matrix between server performance and business metrics.
- Temporal trends of key business metrics.
- Profiles for each cluster.
- Best and worst performing clusters.
- Optimization recommendations and business insights.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
