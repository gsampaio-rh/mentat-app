import pandas as pd
import numpy as np
import datetime


def generate_time_series_data(start_date, periods, freq="T"):
    """
    Generates a time series with the given frequency.

    Args:
    - start_date (datetime): The start date and time for the series.
    - periods (int): Number of periods to generate.
    - freq (str): Frequency string (e.g., 'T' for minutes).

    Returns:
    - pd.DatetimeIndex: Generated time series.
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def simulate_netflix_metrics(time_series):
    """
    Simulates Netflix server metrics for the given time series.

    Args:
    - time_series (pd.DatetimeIndex): Time series for which to generate metrics.

    Returns:
    - pd.DataFrame: DataFrame containing the simulated metrics.
    """
    np.random.seed(42)
    data = {
        "Timestamp": time_series,
        "CPU Utilization (%)": np.random.normal(
            loc=70, scale=10, size=len(time_series)
        ),
        "Memory Utilization (%)": np.random.normal(
            loc=60, scale=10, size=len(time_series)
        ),
        "Network I/O Throughput (Mbps)": np.random.normal(
            loc=40000, scale=10000, size=len(time_series)
        ),
        "Disk I/O Throughput (MB/s)": np.random.normal(
            loc=150, scale=50, size=len(time_series)
        ),
        "Server Configuration": np.random.choice(
            [
                "EC2 API Servers (m5.large)",
                "EC2 Database Servers (r5.large)",
                "EC2 Streaming Servers (c5.large)",
                "EC2 Recommendation System Servers (p3.2xlarge)",
            ],
            size=len(time_series),
        ),
    }
    return pd.DataFrame(data)


def generate_weekly_data():
    """
    Generates one week of simulated Netflix server metrics data with minute-level granularity.

    Returns:
    - pd.DataFrame: DataFrame containing the weekly metrics.
    """
    start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    periods = 7 * 24 * 60  # One week of minute-level data
    time_series = generate_time_series_data(start_date, periods)
    weekly_data = simulate_netflix_metrics(time_series)
    return weekly_data


# Generate the data
weekly_data = generate_weekly_data()

# Save to CSV
csv_file_path = "netflix_weekly_metrics.csv"
weekly_data.to_csv(csv_file_path, index=False)

print(f"CSV file saved as {csv_file_path}")
