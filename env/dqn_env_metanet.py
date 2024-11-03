# Import MetaNet environment and any required utilities
from .metanet_env import MetanetEnv  # Import MetaNet model
import numpy as np
from gym import spaces

class DqnEnvMetaNet:
    
    def min_max_scale(self, x, feature):
        return (x - self.min_max[feature][0]) / (self.min_max[feature][1] - self.min_max[feature][0])

    def __init__(self, m, p=None):
        self.mode = {"train": False, "observe": False, "play": False, m: True}
        self.player = p if self.mode["play"] else None

        # Use MetaNetEnv based on the mode
        if self.mode["train"]:
            print(f"I'm using MetaNetEnv for training")
            self.metanet_env = MetanetEnv()
        elif self.mode["observe"]:
            print(f"I'm using MetaNetEnv for observing")
            self.metanet_env = MetanetEnv()
        
         # """CHANGE FEATURE SCALING HERE""" ############################################################################
        self.min_max = {
        }
        ################################################################################################################


        # Ensure action space is an integer
        self.action_space_n = self.metanet_env.action_space_n  # Ensure it's an integer

        # Ensure observation space is a tuple of integers
        self.observation_space_n = self.metanet_env.observation_space_n  # Ensure it's a tuple of integers
        
        self.ramp_edges = self.find_ramp_edges()
        self.edges_after_ramps = self.find_edges_after_ramps()

    def obs(self):
        obs = self.metanet_env.reset() if not hasattr(self, 'last_obs') else self.last_obs
        print(f"observations are :{obs}")
        return obs

    def rew(self):
        # Calculate the reward after each step
        rew = self.metanet_env.step(self.last_action)[1]
        print(f"reward is : {rew}")
        return rew

    def done(self):
        # Check if the episode is done
        done = self.metanet_env.step(self.last_action)[2]
        return done

    def info(self):
        # Optional: return additional info
        info = {}  # or self.metanet_env.step(self.last_action)[3] if MetaNetEnv provides additional info
        return info

    def reset(self):
        # Reset the environment
        self.last_obs = self.metanet_env.reset()

    def step(self, action):
        # Perform an action and update the environment
        self.last_action = action
        self.last_obs, _, _, _ = self.metanet_env.step(action)

    def reset_render(self):
        pass

    def step_render(self):
        # Visualize the environment's current state
        self.metanet_env.render()

    def save_data_to_pickle(self, file_name):
        self.metanet_env.save_data_to_pickle(file_name)

    def log_info(self):
        # Log information
        print(f"Logging info: {self.info()}")
    
    def find_ramp_edges(self):
        # This method would need to be adapted to work with MetaNetEnv
        return []  # Placeholder; implement based on MetaNetEnv's structure

    def find_edges_after_ramps(self):
        # This method would need to be adapted to work with MetaNetEnv
        return []  # Placeholder; implement based on MetaNetEnv's structure
