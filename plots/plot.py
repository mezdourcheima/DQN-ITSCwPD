import pickle
import os
import matplotlib.pyplot as plt

# Load the data
with open('/Users/cheimamezdour/Projects/PFE/DQN-ITSCwPD/training_data.pkl', 'rb') as f:
    data_over_time = pickle.load(f)


# Create directory if it doesn't exist
plots_dir = '/Users/cheimamezdour/Projects/PFE/DQN-ITSCwPD/plots/'
os.makedirs(plots_dir, exist_ok=True)

# Plot density over time
plt.figure(figsize=(10, 6))
for i, density in enumerate(data_over_time['density']):
    plt.plot(density, label=f'Step {i}')
plt.xlabel('Road Segment')
plt.ylabel('Density (vehicles/km)')
plt.title('Traffic Density Over Time')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'density_over_time.png'))
plt.close()

# Plot flow over time
plt.figure(figsize=(10, 6))
for i, flow in enumerate(data_over_time['flow']):
    plt.plot(flow, label=f'Step {i}')
plt.xlabel('Road Segment')
plt.ylabel('Flow (vehicles/hour)')
plt.title('Traffic Flow Over Time')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'flow_over_time.png'))
plt.close()


# Plot ramp queue length over time
plt.figure(figsize=(10, 6))
for i, queue_length in enumerate(data_over_time['ramp_queue_length']):
    plt.plot(queue_length, label=f'Step {i}')
plt.xlabel('Road Segment')
plt.ylabel('Ramp Queue Length (vehicles)')
plt.title('Ramp Queue Length Over Time')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'ramp_queue_length_over_time.png'))
plt.close()


# Plot speed over time
plt.figure(figsize=(10, 6))
for i, speed in enumerate(data_over_time['speed']):
    plt.plot(speed, label=f'Step {i}')
plt.xlabel('Road Segment')
plt.ylabel('Speed (km/h)')
plt.title('Speed Over Time')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'speed_over_time.png'))
plt.close()

# Plot reward over time
plt.figure(figsize=(10, 6))
for i, reward in enumerate(data_over_time['reward']):
    plt.plot(speed, label=f'Step {i}')
plt.xlabel('Road Segment')
plt.ylabel('Reward')
plt.title('Reward Over Time')
plt.legend()
#plt.show()
plt.savefig(os.path.join(plots_dir, 'reward_over_time.png'))
plt.close()
