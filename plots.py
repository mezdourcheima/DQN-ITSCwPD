import pickle
import matplotlib.pyplot as plt

# Load the data from the pickle file
with open("training_data.pkl", "rb") as f:
    data_over_time = pickle.load(f)

# Plot the metrics over time
plt.figure(figsize=(12, 8))

# Plot density
plt.subplot(2, 2, 1)
for i in range(len(data_over_time['density'][0])):
    plt.plot([density[i] for density in data_over_time['density']], label=f'Segment {i+1}')
plt.title('Density Over Time')
plt.xlabel('Time Step')
plt.ylabel('Density (vehicles/km)')
plt.legend()

# Plot flow
plt.subplot(2, 2, 2)
for i in range(len(data_over_time['flow'][0])):
    plt.plot([flow[i] for flow in data_over_time['flow']], label=f'Segment {i+1}')
plt.title('Flow Over Time')
plt.xlabel('Time Step')
plt.ylabel('Flow (vehicles/h)')
plt.legend()

# Plot ramp queue length
plt.subplot(2, 2, 3)
for i in range(len(data_over_time['ramp_queue_length'][0])):
    plt.plot([ramp_queue_length[i] for ramp_queue_length in data_over_time['ramp_queue_length']], label=f'Segment {i+1}')
plt.title('Ramp Queue Length Over Time')
plt.xlabel('Time Step')
plt.ylabel('Queue Length (vehicles)')
plt.legend()

# Plot reward
plt.subplot(2, 2, 4)
plt.plot(data_over_time['reward'])
plt.title('Reward Over Time')
plt.xlabel('Time Step')
plt.ylabel('Reward')

plt.tight_layout()
plt.show()
