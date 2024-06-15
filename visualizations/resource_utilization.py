import matplotlib.pyplot as plt
from utils.utils import save_plot
import logging

def plot_resource_utilization_efficiency(avg_utilization_df):
    """
    Plot resource utilization efficiency.

    Args:
    - avg_utilization_df (pd.DataFrame): DataFrame containing average utilization data.
    """
    try:
        high_utilization_threshold = avg_utilization_df[['CPU Utilization (%)', 'Memory Utilization (%)']].quantile(0.75)
        low_utilization_threshold = avg_utilization_df[['CPU Utilization (%)', 'Memory Utilization (%)']].quantile(0.25)

        avg_utilization_df['Color'] = 'blue'
        avg_utilization_df.loc[(avg_utilization_df['CPU Utilization (%)'] >= high_utilization_threshold['CPU Utilization (%)']) &
                               (avg_utilization_df['Memory Utilization (%)'] >= high_utilization_threshold['Memory Utilization (%)']), 'Color'] = 'red'
        avg_utilization_df.loc[(avg_utilization_df['CPU Utilization (%)'] <= low_utilization_threshold['CPU Utilization (%)']) &
                               (avg_utilization_df['Memory Utilization (%)'] <= low_utilization_threshold['Memory Utilization (%)']), 'Color'] = 'green'

        fig = plt.figure(figsize=(10, 6))
        for i in range(len(avg_utilization_df)):
            plt.scatter(avg_utilization_df['CPU Utilization (%)'][i], avg_utilization_df['Memory Utilization (%)'][i], color=avg_utilization_df['Color'][i])
            plt.text(avg_utilization_df['CPU Utilization (%)'][i], avg_utilization_df['Memory Utilization (%)'][i], 
                     avg_utilization_df['Server Configuration'][i], fontsize=9)

        plt.scatter([], [], color='red', label='High Utilization (Red)')
        plt.scatter([], [], color='green', label='Low Utilization (Green)')
        plt.legend(loc='upper right')

        plt.title('Resource Utilization Efficiency')
        plt.xlabel('Average CPU Utilization (%)')
        plt.ylabel('Average Memory Utilization (%)')
        plt.grid(True)
        save_plot(fig, "resource_utilization_efficiency.png")
        plt.show()
        logging.info("Resource utilization efficiency plotted successfully.")
    except Exception as e:
        logging.error(f"An error occurred while plotting resource utilization efficiency: {e}")
