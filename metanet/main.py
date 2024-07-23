# main.py
from env.dqn_env import DqnEnv
from MetaDQNAgent import MetaDQNAgent
import numpy as np

def main():
    # Initialize the environment and the agent
    m = "train"  # Assuming "train" is the mode you want to set
    env = DqnEnv(m)
    state_size = env.observation_space_n
    action_size = env.action_space_n
    agent = MetaDQNAgent(state_size, action_size)
    
    # Parameters
    num_episodes = 1000
    batch_size = 32
    
    # Train the agent
    agent.train(env, num_episodes, batch_size)

if __name__ == "__main__":
    main()
