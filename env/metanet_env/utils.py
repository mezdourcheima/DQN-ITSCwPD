import numpy as np

metanet_params = {
    'num_links': 5,  # Number of road segments
    'link_lengths': np.array([1.0, 1.2, 1.5, 0.8, 1.3]),  # Lengths of each road segment in km
    'free_flow_speed': 30.0,  # Free-flow speed in km/h for all segments
    'time_step': 1,  # Time step for the simulation in minutes
    'alpha': 0.5,  # Anticipation factor for speed update equation
    'rho_critical': 30.0,  # Critical density in vehicles per km
    'beta': 1.0,  # Speed adaptation parameter
    'max_density': 100.0,  # Maximum vehicle density in vehicles per km
    'jam_density': 150.0,  # Jam density in vehicles per km
    'inflow_rate': 200,  # Inflow rate at the network entrance in vehicles per hour
    'outflow_rate': 150  # Outflow rate at the network exit in vehicles per hour
}