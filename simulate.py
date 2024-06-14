import pandas as pd
import numpy as np


# Function to simulate performance metrics
def simulate_metrics(num_samples, config_name):
    np.random.seed(42)  # For reproducibility
    data = {
        "Server Configuration": [config_name] * num_samples,
        "CPU Utilization (%)": np.random.normal(loc=70, scale=10, size=num_samples),
        "Memory Utilization (%)": np.random.normal(loc=50, scale=5, size=num_samples),
        "Network I/O Throughput (Mbps)": np.random.normal(
            loc=30000, scale=5000, size=num_samples
        ),
        "Disk I/O Throughput (MB/s)": np.random.normal(
            loc=100, scale=20, size=num_samples
        ),
    }
    return pd.DataFrame(data)


# Simulate data for different server configurations
num_samples = 1000
configurations = ["Server A", "Server B", "Server C"]

simulated_data = pd.concat(
    [simulate_metrics(num_samples, config) for config in configurations],
    ignore_index=True,
)

# Save the simulated data to a CSV file
csv_file_path = "/mnt/data/simulated_server_metrics.csv"
simulated_data.to_csv(csv_file_path, index=False)

import ace_tools as tools

tools.display_dataframe_to_user(
    name="Simulated Server Metrics", dataframe=simulated_data
)
