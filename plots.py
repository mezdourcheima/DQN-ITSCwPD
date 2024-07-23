import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/Users/cheimamezdour/Projects/PFE/DQN-ITSCwPD/logs/test/data/1tls_3x3.csv'
data = pd.read_csv(file_path)

# Ensure the output directory exists
output_dir = '/Users/cheimamezdour/Projects/PFE/DQN-ITSCwPD/logs/test/plots'
os.makedirs(output_dir, exist_ok=True)

# Function to plot and save the metrics
def plot_and_save_metrics(data, metrics, xlabel, ylabel, title, output_dir):
    plt.figure(figsize=(15, 10))
    for metric in metrics:
        plt.plot(data['ep'], data[metric], label=metric)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{title}.png')
    plt.savefig(output_path)
    plt.close()

# Metrics to plot
metrics = [
    'r', 'ctrl_con_p_rate', 'sum_delay', 'sum_waiting_time',
    'avg_acc_waiting_time', 'avg_queue_length', 'total_density',
    'total_flow', 'total_ramp_queue_length'
]

# Plotting and saving the metrics per episode
plot_and_save_metrics(data, metrics, xlabel='Episode', ylabel='Value', title='Metrics per Episode', output_dir=output_dir)
