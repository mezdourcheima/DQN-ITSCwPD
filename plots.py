import argparse
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dqn.agent import DQNAgent
from dqn.env_wrap import CustomEnvWrapper  # Assuming the class is in custom_env_wrapper.py
from env import CustomEnv  # Import CustomEnv from the correct module

def collect_data(env, step, total_reward):
    info = env._info()
    return {
        "info" : info,
        "step": step,
        "total_Density": info.get("total_Density", 0),  # Replace with actual method to get total density
        "queue_length": info.get("queue_length", 0),  # Replace with actual method to get queue length
        "reward_total": total_reward,
        "flow": info.get("flow", 0),  # Replace with actual method to get flow
        "penalty": info.get("penalty", 0)  # Replace with actual method to get penalty
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observe a trained RL agent")
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_csv', type=str, default='observation_data.csv', help='Path to save the output CSV file')
    args = parser.parse_args()

    # Load the trained model
    agent = DQNAgent.load(args.model)

    # Set up the environment
    env = CustomEnvWrapper(CustomEnv())
    
    # Initialize data collection
    collected_data = []

    # Observation loop
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Collect data
        data = collect_data(env, step, total_reward)
        collected_data.append(data)
        
        state = next_state
        step += 1

    # Save collected data to CSV
    csv_columns = ["step", "total_Density", "queue_length", "reward_total", "flow", "penalty"]
    csv_file = args.output_csv

    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in collected_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    print(f"Observation data saved to {csv_file}")

    # Plotting the data
    df = pd.DataFrame(collected_data)
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(df['step'], df['total_Density'])
    plt.title('Total Density over Steps')

    plt.subplot(3, 2, 2)
    plt.plot(df['step'], df['queue_length'])
    plt.title('Queue Length over Steps')

    plt.subplot(3, 2, 3)
    plt.plot(df['step'], df['reward_total'])
    plt.title('Total Reward over Steps')

    plt.subplot(3, 2, 4)
    plt.plot(df['step'], df['flow'])
    plt.title('Flow over Steps')

    plt.subplot(3, 2, 5)
    plt.plot(df['step'], df['penalty'])
    plt.title('Penalty over Steps')

    plt.tight_layout()
    plt.show()
