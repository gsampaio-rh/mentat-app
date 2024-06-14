import pandas as pd
import numpy as np
import datetime


def generate_time_series_data(start_date, periods, freq="T"):
    return pd.date_range(start=start_date, periods=periods, freq=freq)


def simulate_random_events(data, event_prob=0.001, spike_magnitude_range=(1.5, 3)):
    for column in [
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network I/O Throughput (Mbps)",
        "Disk I/O Throughput (MB/s)",
    ]:
        event_indices = np.random.choice(
            [False, True], size=len(data), p=[1 - event_prob, event_prob]
        )
        spike_magnitudes = np.random.uniform(
            spike_magnitude_range[0], spike_magnitude_range[1], size=event_indices.sum()
        )
        data.loc[event_indices, column] *= (
            np.random.choice([1, -1], size=event_indices.sum()) * spike_magnitudes
        )
        data[column] = np.clip(data[column], 0, None)

    return data


def simulate_netflix_metrics(time_series):
    np.random.seed(42)
    minutes_in_day = 24 * 60
    days_in_week = 7

    daily_pattern = np.sin(
        2 * np.pi * (np.arange(len(time_series)) % minutes_in_day) / minutes_in_day
    )
    weekly_pattern = np.sin(
        2
        * np.pi
        * (np.arange(len(time_series)) % (minutes_in_day * days_in_week))
        / (minutes_in_day * days_in_week)
    )

    data = {
        "Timestamp": time_series,
        "CPU Utilization (%)": np.clip(
            70
            + 10 * np.random.normal(size=len(time_series))
            + 5 * daily_pattern
            + 2 * weekly_pattern,
            0,
            100,
        ),
        "Memory Utilization (%)": np.clip(
            60
            + 10 * np.random.normal(size=len(time_series))
            + 4 * daily_pattern
            + 1.5 * weekly_pattern,
            0,
            100,
        ),
        "Network I/O Throughput (Mbps)": np.clip(
            40000
            + 10000 * np.random.normal(size=len(time_series))
            + 3000 * daily_pattern
            + 1000 * weekly_pattern,
            0,
            None,
        ),
        "Disk I/O Throughput (MB/s)": np.clip(
            150
            + 50 * np.random.normal(size=len(time_series))
            + 20 * daily_pattern
            + 10 * weekly_pattern,
            0,
            None,
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

    df = pd.DataFrame(data)
    df = simulate_random_events(df)
    return df


def simulate_business_metrics(df):
    np.random.seed(42)

    # Simulate business metrics
    business_data = {
        "Timestamp": df["Timestamp"],
        "Server Configuration": df["Server Configuration"],
        "Response Time (ms)": np.clip(
            200
            + 0.1 * df["CPU Utilization (%)"]
            + 0.1 * df["Memory Utilization (%)"]
            + np.random.normal(scale=20, size=len(df)),
            0,
            None,
        ),
    }
    business_df = pd.DataFrame(business_data)

    business_df["Customer Satisfaction (CSAT)"] = np.clip(
        90
        - 0.1 * business_df["Response Time (ms)"]
        + np.random.normal(scale=5, size=len(business_df)),
        0,
        100,
    )
    business_df["Operational Costs ($)"] = (
        5000
        + 0.05 * df["Network I/O Throughput (Mbps)"]
        + np.random.normal(scale=500, size=len(business_df))
    )
    business_df["Service Uptime (%)"] = np.clip(
        99.9
        - 0.01 * (df["CPU Utilization (%)"] > 80).astype(int)
        + np.random.normal(scale=0.1, size=len(business_df)),
        0,
        100,
    )

    return business_df


def generate_weekly_data():
    start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    periods = 7 * 24 * 60  # One week of minute-level data
    time_series = generate_time_series_data(start_date, periods)
    operational_data = simulate_netflix_metrics(time_series)
    business_data = simulate_business_metrics(operational_data)
    return operational_data, business_data


# Generate the data
operational_data, business_data = generate_weekly_data()

# Save to CSV
operational_csv_file_path = "data/netflix_operational_metrics.csv"
business_csv_file_path = "data/netflix_business_metrics.csv"
operational_data.to_csv(operational_csv_file_path, index=False)
business_data.to_csv(business_csv_file_path, index=False)
