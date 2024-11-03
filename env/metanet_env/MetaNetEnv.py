from .control import dynamic_ramp_metering, dynamic_speed_control
import numpy as np
import gym
import torch
from gym import spaces
from .network import initialize_network, update_network
from .control import ramp_metering, calculate_ramp_queue_length
from env import xml_parser as ps
import matplotlib.pyplot as plt
import pickle


class MetanetEnv(gym.Env):
    def __init__(self):
        super(MetanetEnv, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.length = 10  # Length of the road (km)
        print(f"Road length: {self.length} km")
        
        self.dx = 1.0  # Space step (km)
        print(f"Space step: {self.dx} km")
        
        self.dt = 0.1  # Time step (seconds)
        print(f"Time step: {self.dt} seconds")
        
        self.nx = int(self.length / self.dx)

        self.time_steps = 200  # Number of time steps
        print(f"Number of time steps: {self.time_steps}")
        
        # Initialize network state
        self.rho, self.nx, self.lanes, self.rho_max, self.v_free = initialize_network(self.length, self.dx)
        print(f"Initial density: {self.rho}")
        print(f"Number of segments: {self.nx}")
        print(f"Number of lanes: {self.lanes}")
        print(f"Maximum density (rho_max): {self.rho_max}")
        print(f"Free flow speed (v_free): {self.v_free} km/h")

        self.alpha = 1.0  # Speed-density relationship parameter
        print(f"Speed-density relationship parameter (alpha): {self.alpha}")
        
        self.ramp_index = 2  # Example index for ramp metering
        print(f"Ramp metering index: {self.ramp_index}")

        self.queue_length = np.zeros(self.nx)  # Queue length for each segment
        
        # Initialize storage for metrics over time
        self.data_over_time = {
                    'density': [],
                    'flow': [],
                    'ramp_queue_length': [],
                    'speed': [],
                    'reward': []
                    
        }
        # Initialize sum_delay_min and sum_waiting_time_min
        self.sum_delay_min = float('inf')  # Set to infinity initially
        self.sum_waiting_time_min = float('inf')  # Set to infinity initially
        
        save_interval=1000
        # Pickle saving interval and counter
        self.save_interval = save_interval
        self.target_density = np.random.uniform(self.rho_max * 0.3, self.rho_max * 0.7)  # Dynamic target between 30% and 70% of max density

        self.step_count = 0

        self.speed_limit_index = 3  # Example index for speed limit control
        self.max_speed_limit = 120  # Maximum speed limit (km/h)
        self.min_speed_limit = 60  # Minimum speed limit (km/h)
        self.max_queue_length = 100  # Example maximum queue length
        self.max_metering_rate = 0.5  # Maximum metering rate (vehicles/second)
        
        self.dtse_shape = self.get_dtse_shape()
        print(f"Observation space shape (dtse_shape): {self.dtse_shape}")
        
        # Define action space: 3 discrete actions (example)
        self.action_space_n = 3
        print(f"Action space: {self.action_space_n}")
        
        # Define observation space
        self.observation_space_n = self.dtse_shape
        print(f"Observation space: {self.observation_space_n}")

        # For rendering
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(0, self.rho_max)
        self.ax.set_xlim(0, self.length)
        self.ax.set_xlabel('Position on road (km)')
        self.ax.set_ylabel('Traffic density (vehicles/km)')
        self.ax.set_title('Traffic Density Distribution')
        print("Rendering setup complete.")

    def get_sum_delay_a_sum_waiting_time(self):
        """Calculate total delay and waiting time for the current state of the network."""
        total_delay = 0
        total_waiting_time = 0

        # Loop through each segment in the road network
        for segment in range(self.nx):
            # Calculate actual speed in the segment based on density
            actual_speed = self.v_free * (1 - self.alpha * self.rho[segment] / self.rho_max)

            # Free-flow travel time (when traffic is at free-flow speed)
            free_flow_travel_time = self.dx / self.v_free

            # Actual travel time based on current traffic conditions
            if actual_speed > 0:
                actual_travel_time = self.dx / actual_speed
            else:
                actual_travel_time = float('inf')  # Handle low or zero speeds
            
            # Avoid using inf for delay
            delay = actual_travel_time - free_flow_travel_time
            if not np.isinf(delay):
                total_delay += max(0, delay)  # Only positive delays are counted

            # Waiting time: consider any speed below a threshold as "waiting"
            waiting_time = 0
            if actual_speed < self.min_speed_limit:
                waiting_time = self.dt  # Time step duration is the waiting time
            total_waiting_time += waiting_time

        print(f"Total Delay: {total_delay}, Total Waiting Time: {total_waiting_time}")
        return total_delay, total_waiting_time




    def check_for_collisions(self):
        """Check for collisions based on traffic density and return the number of collisions."""
        collision_count = 0

        # Define a critical density threshold
        critical_density = 0.9 * self.rho_max  # 90% of the maximum allowable density

        # Loop through each road segment
        for segment in range(self.nx):
            density = self.rho[segment]  # Current density at the segment
            
            # If density exceeds the critical threshold, count as a collision
            if density > critical_density:
                collision_count += 1
                print(f"Collision detected at segment {segment} with density {density}")

        # Store the collision count for use in reward calculation
        self.collision_count = collision_count

        return collision_count
    
    def calculate_reward(self, rho, flow, target_density, rho_max):
        total_rew = 0

        # Get the delay and waiting time metrics
        sum_delay, sum_waiting_time = self.get_sum_delay_a_sum_waiting_time()

        # Ensure sum_delay_min and sum_waiting_time_min are valid (non-zero and non-inf)
        self.sum_delay_min = min(self.sum_delay_min, sum_delay) if sum_delay > 0 else 1e-6
        self.sum_waiting_time_min = min(self.sum_waiting_time_min, sum_waiting_time) if sum_waiting_time > 0 else 1e-6

        rew_delay = 0 if self.sum_delay_min == 0 else 1 + sum_delay / self.sum_delay_min
        rew_waiting_time = 0 if self.sum_waiting_time_min == 0 else 1 + sum_waiting_time / self.sum_waiting_time_min

        # Weighting factors for delay and waiting time rewards
        w1, w2 = 1.0, 1.0

        rew = w1 * rew_delay + w2 * rew_waiting_time

        # Normalize reward to range [0, 1]
        normalized_reward = np.clip(rew / 1e4, 0, 1)  # Adjust 1e4 based on typical values

        # Check for collisions and apply a smaller penalty
        self.check_for_collisions()  # Update self.collision_count
        collision_penalty = -2 * self.collision_count  # Reduced penalty

        total_rew += collision_penalty

        # Combine the total reward
        final_rew = np.clip(normalized_reward, 0, 1) + total_rew + 50 


        # Avoid returning NaN values
        final_rew = np.nan_to_num(final_rew, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Final total reward: {final_rew}")
        
        return final_rew

            

    def step(self, action):
        self.step_count += 1
        print(f"Action received: {action}")
        
        # Calculate flow
        q = self.rho * self.v_free * (1 - self.alpha * self.rho / self.rho_max)
        print(f"Initial Flow calculated (q): {q}")
        
        # Clamp flow to prevent extreme values (you can adjust these limits)
        q = np.clip(q, 0, 1e6)
        print(f"Clamped Flow (q): {q}")
        
        # Update the network state based on flow
        self.rho = update_network(self.rho, q, self.dt, self.dx)
        print(f"Updated density (rho) after network update: {self.rho}")
        
        # Clamp density to prevent extreme values
        self.rho = np.clip(self.rho, 0, self.rho_max)
        
        # Update ramp queue length
        incoming_vehicles = np.random.randint(0, 10)  # Example incoming vehicle rate
        self.queue_length[self.ramp_index] = calculate_ramp_queue_length(self.queue_length[self.ramp_index], incoming_vehicles, self.max_queue_length)
        
        # Apply dynamic ramp metering based on action
        if action == 1:  # Ramp metering action
            self.rho, self.queue_length[self.ramp_index] = dynamic_ramp_metering(
                self.rho, self.ramp_index, self.dt, self.queue_length[self.ramp_index],
                self.max_metering_rate, target_density=self.rho_max / 2)
        
        # Apply dynamic speed limits
        elif action == 2:
            self.v_free = dynamic_speed_control(
                self.rho, self.v_free, self.min_speed_limit, self.max_speed_limit,
                target_density=self.rho_max / 2)
            
        # Calculate the reward using the defined function
        reward = self.calculate_reward(self.rho, q, self.target_density, self.rho_max)

        # Debugging: Print out the reward and other relevant information
        print(f"Action: {action}, Reward: {reward}, Density: {self.rho}, Flow: {q}")
    
        # Example calculations for each metric
        speed = self.v_free * (1 - self.alpha * self.rho / self.rho_max)
        print(f"Speed calculated: {speed}")
        
        # Clamp speed to prevent extreme values
        speed = np.clip(speed, 0, 150)  # Assuming 150 km/h as a reasonable upper limit
        

    
 
        # Store the current state in the data_over_time dictionary
        self.data_over_time['density'].append(self.rho.copy())
        self.data_over_time['flow'].append(q.copy())
        self.data_over_time['ramp_queue_length'].append(self.queue_length.copy())
        self.data_over_time['speed'].append(speed.copy())
        self.data_over_time['reward'].append(reward.copy())

        
        # Save the data periodically
        if self.step_count % self.save_interval == 0:
            self.save_data_to_pickle("training_data.pkl")

        flow = q  # Flow is directly from the initial calculation
        ramp_queue_length = self.queue_length
        print(f"Ramp queue length: {ramp_queue_length}")

        # Create observation
        density = np.tile(self.rho[:, np.newaxis], (1, self.dtse_shape[2]))
        flow = np.tile(flow[:, np.newaxis], (1, self.dtse_shape[2]))
        ramp_queue_length = np.tile(ramp_queue_length[:, np.newaxis], (1, self.dtse_shape[2]))
        speed = np.tile(speed[:, np.newaxis], (1, self.dtse_shape[2]))

        print(f"Density matrix: {density}")
        print(f"Flow matrix: {flow}")
        print(f"Ramp queue length matrix: {ramp_queue_length}")
        print(f"Speed matrix: {speed}")
        
        # Combine the observations
        observation = np.array([density, flow, ramp_queue_length, speed])
        observation = observation.reshape(self.dtse_shape)  # Ensure the shape matches the expected input
        print(f"Observation combined and reshaped: {observation.shape}")
    

        
        
        # Example termination condition (you can define your own)
        done = False
        
        return observation, reward, done, {}
    

    def apply_variable_speed_limit(self):
        """
        Apply variable speed limits based on current traffic conditions.
        This function adjusts the speed limit dynamically to smooth traffic flow.
        """
        print("Applying variable speed limits.")
        # Simple logic to decrease speed limit when density is high
        if np.mean(self.rho) > self.rho_max / 2:
            new_speed_limit = self.v_free * 0.8  # Reduce speed limit by 20%
            self.v_free = max(self.min_speed_limit, new_speed_limit)
        else:
            self.v_free = min(self.v_free * 1.1, self.max_speed_limit)  # Gradually restore speed limit
        print(f"New speed limit (v_free): {self.v_free} km/h")
    
    def save_data_to_pickle(self, filename):
        """Save the collected data over time to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.data_over_time, f)
        print(f"Data saved to {filename} after step {self.step_count}")


    def reset(self):
        print("Resetting environment.")
        
        self.rho, self.nx, self.lanes, self.rho_max, self.v_free = initialize_network(self.length, self.dx)
        print(f"Initial density after reset: {self.rho}")
        
        speed = self.v_free * (1 - self.alpha * self.rho / self.rho_max)
        print(f"Initial speed after reset: {speed}")
        
        flow = self.rho * speed
        print(f"Initial flow after reset: {flow}")
        
        ramp_queue_length = self.queue_length  # Initial ramp queue length
        print(f"Initial ramp queue length after reset: {ramp_queue_length}")

        # Create initial observation
        density = np.tile(self.rho[:, np.newaxis], (1, self.dtse_shape[2]))
        flow = np.tile(flow[:, np.newaxis], (1, self.dtse_shape[2]))
        ramp_queue_length = np.tile(ramp_queue_length[:, np.newaxis], (1, self.dtse_shape[2]))
        speed = np.tile(speed[:, np.newaxis], (1, self.dtse_shape[2]))
        print(f"Initial observation density matrix: {density}")
        print(f"Initial observation flow matrix: {flow}")
        print(f"Initial observation ramp queue length matrix: {ramp_queue_length}")
        print(f"Initial observation speed matrix: {speed}")

        observation = np.array([density, flow, ramp_queue_length, speed])
        observation = observation.reshape(self.dtse_shape)
        print(f"Initial observation combined and reshaped: {observation.shape}")
        
        self.data_over_time = {
            'density': [],
            'flow': [],
            'ramp_queue_length': [],
            'speed': [],
            'reward': []
        }
        return observation
    
    def render(self, mode='human'):
        # Update the plot with the current traffic density
        self.line.set_data(np.linspace(0, self.length, self.nx), self.rho)
        self.ax.set_ylim(0, self.rho_max)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        print("Rendering current traffic density.")
    
    def close(self):
        # Close the plot window when done
        plt.close(self.fig)
        print("Closed the plot window.")

    def get_dtse_shape(self):
        # Define the shape of your observation space here
        shape = (
            4,  # 4 observations: density, flow, ramp queue length, speed
            10,  # Number of road segments (or areas of interest)
            1   # Single spatial cell for simplicity (can be adjusted)
        )
        print(f"Defined dtse_shape: {shape}")
        return shape

    def plot_network(self, save_path=None):
        """
        Plot the current state of the network, including density and potentially other variables.
        Optionally save the plot to a file if a save_path is provided.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, self.length, self.nx), self.rho, label='Traffic Density')
        plt.xlabel('Position on road (km)')
        plt.ylabel('Traffic Density (vehicles/km)')
        plt.title('Traffic Density Distribution along the Road')
        plt.ylim(0, self.rho_max)
        plt.xlim(0, self.length)
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()



