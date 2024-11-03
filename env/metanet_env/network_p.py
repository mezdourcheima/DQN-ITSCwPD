import numpy as np

def initialize_network(length, dx, ramp_position=None):
    """
    Initialize the traffic network.
    
    Parameters:
    - length: Total length of the road (in kilometers).
    - dx: Space step (in kilometers).
    - ramp_position: Optional position of the on-ramp in kilometers.
    
    Returns:
    - rho: Initial traffic density along the road (vehicles/km).
    - nx: Number of segments in the road network.
    - ramp_idx: Index position of the on-ramp in the density array.
    """
    nx = int(length / dx)
    rho = np.zeros(nx)  # Initial density (vehicles/km)
    ramp_idx = None

    # Example initial condition: some congestion in the middle of the road
    rho[int(nx/4):int(nx/2)] = 30  # Initial congestion

    # Determine the ramp position in the network, if provided
    if ramp_position is not None:
        ramp_idx = int(ramp_position / dx)
    
    return rho, nx, ramp_idx

def calculate_flow(rho, v_free, alpha, rho_max):
    """
    Calculate the traffic flow based on current density and traffic parameters.
    
    Parameters:
    - rho: Traffic density along the road (vehicles/km).
    - v_free: Free flow speed (km/h).
    - alpha: Speed-density relationship parameter.
    - rho_max: Maximum traffic density (vehicles/km).
    
    Returns:
    - q: Traffic flow along the road (vehicles/hour).
    """
    speed = v_free * (1 - alpha * (rho / rho_max))
    q = rho * speed
    return q

def ramp_metering(rho, meter_rate, ramp_idx):
    """
    Apply ramp metering to control the inflow from an on-ramp.
    
    Parameters:
    - rho: Current traffic density along the road (vehicles/km).
    - meter_rate: Rate at which vehicles are allowed to enter the highway from the ramp.
    - ramp_idx: Index position of the on-ramp in the density array.
    
    Returns:
    - rho: Updated traffic density after ramp metering.
    """
    if ramp_idx is not None:
        rho[ramp_idx] += meter_rate  # Simple additive model for ramp metering
    return rho

def update_network(rho, v_free, alpha, rho_max, dt, dx, inflow=0, outflow=0):
    """
    Update the traffic network based on the flow and current density.
    
    Parameters:
    - rho: Current traffic density along the road (vehicles/km).
    - v_free: Free flow speed (km/h).
    - alpha: Speed-density relationship parameter.
    - rho_max: Maximum traffic density (vehicles/km).
    - dt: Time step (in seconds).
    - dx: Space step (in kilometers).
    - inflow: Vehicles entering the network (vehicles/hour).
    - outflow: Vehicles leaving the network (vehicles/hour).
    
    Returns:
    - rho: Updated traffic density along the road.
    """
    q = calculate_flow(rho, v_free, alpha, rho_max)
    rho[1:] = rho[1:] - dt/dx * (q[1:] - q[:-1])

    # Apply boundary conditions for inflow and outflow
    rho[0] = rho[0] + inflow * dt/dx
    rho[-1] = max(0, rho[-1] - outflow * dt/dx)

    return rho

def apply_ramp_metering(rho, ramp_idx, inflow_rate, v_free, alpha, rho_max, dt, dx):
    """
    Apply ramp metering logic within the network update process.
    
    Parameters:
    - rho: Current traffic density along the road (vehicles/km).
    - ramp_idx: Index position of the on-ramp in the density array.
    - inflow_rate: Rate at which vehicles are allowed to enter the highway from the ramp.
    - v_free: Free flow speed (km/h).
    - alpha: Speed-density relationship parameter.
    - rho_max: Maximum traffic density (vehicles/km).
    - dt: Time step (in seconds).
    - dx: Space step (in kilometers).
    
    Returns:
    - rho: Updated traffic density along the road.
    """
    # Calculate the flow with ramp metering applied
    q = calculate_flow(rho, v_free, alpha, rho_max)
    
    # Apply the ramp metering
    if ramp_idx is not None:
        rho[ramp_idx] += inflow_rate * dt/dx

    # Update the density with the new flow
    rho = update_network(rho, v_free, alpha, rho_max, dt, dx)
    
    return rho
