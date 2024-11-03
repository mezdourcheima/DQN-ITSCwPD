# control.py
import numpy as np

def ramp_metering(rho, ramp_index, dt, queue_length, max_metering_rate):
    """
    Apply ramp metering logic and manage queue lengths.
    
    Parameters:
    - rho: Current traffic density along the road (vehicles/km).
    - ramp_index: Index of the road segment where the ramp metering is applied.
    - dt: Time step (in seconds).
    - queue_length: Current queue length at the ramp.
    - max_metering_rate: Maximum metering rate allowed (vehicles/second).
    
    Returns:
    - rho: Updated traffic density along the road.
    - queue_length: Updated queue length at the ramp.
    """
    if queue_length > 0:
        # Release vehicles onto the main road based on metering rate
        released_vehicles = min(queue_length, max_metering_rate * dt)
        rho[ramp_index] += released_vehicles / dt
        queue_length -= released_vehicles
    return rho, queue_length

def calculate_ramp_queue_length(queue_length, incoming_vehicles, max_queue_length):
    """
    Update ramp queue length based on incoming vehicles and metering rate.
    
    Parameters:
    - queue_length: Current queue length at the ramp.
    - incoming_vehicles: Number of vehicles arriving at the ramp.
    - max_queue_length: Maximum queue length allowed.
    
    Returns:
    - queue_length: Updated queue length at the ramp.
    """
    queue_length = min(queue_length + incoming_vehicles, max_queue_length)
    return queue_length


def dynamic_ramp_metering(rho, ramp_index, dt, queue_length, max_metering_rate, target_density):
    """
    Adjust the ramp metering rate based on real-time traffic density.
    :param rho: Traffic density on the road segments
    :param ramp_index: Index of the ramp
    :param dt: Time step
    :param queue_length: Current queue length at the ramp
    :param max_metering_rate: Maximum allowed metering rate (vehicles per second)
    :param target_density: Desired target density to maintain
    :return: Updated density and queue length
    """
    # Determine the current density at the segment where the ramp is located
    current_density = rho[ramp_index]

    # Adjust metering rate based on the deviation from target density
    metering_rate = max_metering_rate * (1 - (current_density / target_density))

    # Ensure metering rate is non-negative and does not exceed maximum
    metering_rate = np.clip(metering_rate, 0, max_metering_rate)

    # Update queue length
    queue_length += dt * metering_rate

    # Update the density on the road segment
    rho[ramp_index] += metering_rate * dt / (queue_length + 1e-5)

    # Return updated density and queue length
    return rho, queue_length


def dynamic_speed_control(rho, v_free, min_speed_limit, max_speed_limit, target_density):
    """
    Adjust the speed limit based on real-time traffic density.
    :param rho: Traffic density on the road segments
    :param v_free: Current free flow speed
    :param min_speed_limit: Minimum allowable speed limit (km/h)
    :param max_speed_limit: Maximum allowable speed limit (km/h)
    :param target_density: Desired target density to maintain
    :return: Updated speed limits
    """
    # Calculate the average density
    avg_density = np.mean(rho)

    # Adjust speed limit based on the deviation from target density
    if avg_density > target_density:
        v_free *= 0.8  # Reduce speed limit if density is high
    else:
        v_free *= 1.1  # Gradually increase speed limit if density is low

    # Ensure speed limits are within the allowable range
    v_free = np.clip(v_free, min_speed_limit, max_speed_limit)

    return v_free


def get_sum_delay_a_sum_waiting_time(self):
    total_delay = 0
    total_waiting_time = 0
    
    for segment in range(self.nx):  # Assuming nx is the number of segments
        actual_speed = self.v_free * (1 - self.alpha * self.rho[segment] / self.rho_max)  # Actual speed calculation
        
        # Free flow travel time
        free_flow_travel_time = self.dx / self.v_free
        
        # Actual travel time
        actual_travel_time = self.dx / actual_speed if actual_speed > 0 else float('inf')  # Avoid division by zero
        
        # Delay is the difference between actual travel time and free-flow travel time
        delay = actual_travel_time - free_flow_travel_time
        total_delay += max(0, delay)  # Only consider positive delays

        # Waiting time: if speed is below a threshold, we consider the vehicle to be 'waiting'
        waiting_time = 0
        if actual_speed < self.min_speed_limit:  # Define a speed limit below which vehicles are considered waiting
            waiting_time = self.dt  # Time step duration is the waiting time for this step
        total_waiting_time += waiting_time
    
    return total_delay, total_waiting_time



def check_for_collisions(self):
    """Check for collisions based on vehicle speed and return the number of collisions."""
    collision_count = 0

    for segment in range(self.nx):
        actual_speed = self.v_free * (1 - self.alpha * self.rho[segment] / self.rho_max)

        # If speed drops below a critical threshold, count as a collision
        if actual_speed < self.collision_speed_threshold:
            collision_count += 1
            print(f"Collision detected at segment {segment} due to low speed {actual_speed}")

    return collision_count
