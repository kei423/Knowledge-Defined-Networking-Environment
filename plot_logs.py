import re
import os
import numpy as np
import matplotlib.pyplot as plt

def smooth(scalars, weight=0.85):
    """
    Exponential moving average smoothing for noisy RL plots.
    Similar to TensorBoard smoothing.
    """
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def parse_and_plot(log_filename="mpdrl_simulation.log"):
    if not os.path.exists(log_filename):
        print(f"Error: Log file '{log_filename}' not found.")
        print("Please run the simulation first to generate the logs.")
        return

    # Data containers
    rewards = []
    critic_losses = []
    actor_losses = []

    # Regex patterns based on the logging format in runner.py
    reward_pattern = re.compile(r"Avg Step Reward \(-LU_SD\): ([-.\d]+)")
    critic_pattern = re.compile(r"Critic Update: L1 \(MSE\) Loss = ([-.\d]+)")
    actor_pattern = re.compile(r"Total_Loss = ([-.\d]+)")

    print(f"Parsing '{log_filename}'...")

    with open(log_filename, "r") as f:
        for line in f:
            # Check for Reward
            reward_match = reward_pattern.search(line)
            if reward_match:
                rewards.append(float(reward_match.group(1)))
            
            # Check for Critic Loss
            critic_match = critic_pattern.search(line)
            if critic_match:
                critic_losses.append(float(critic_match.group(1)))
                
            # Check for Actor Loss
            actor_match = actor_pattern.search(line)
            if actor_match:
                actor_losses.append(float(actor_match.group(1)))

    print(f"Found {len(rewards)} episodes, {len(critic_losses)} critic updates, and {len(actor_losses)} actor updates.")

    if not rewards and not critic_losses and not actor_losses:
        print("No valid metrics found in the log file. Make sure training ran and logged properly.")
        return

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.style.use('seaborn-v0_8-darkgrid') # Nice modern grid style
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('MPDRL Training Metrics', fontsize=16, fontweight='bold')

    # 1. Plot Rewards
    if rewards:
        axes[0].plot(rewards, color='blue', marker='o', linestyle='-', alpha=0.7)
        axes[0].set_title("Average Step Reward per Episode ")
        axes[0].set_ylabel("Reward (-LU_SD - Drop Rate)")
        axes[0].set_xlabel("Episode")
    else:
        axes[0].text(0.5, 0.5, 'No Reward Data', ha='center', va='center')

    # 2. Plot Critic Loss
    if critic_losses:
        axes[1].plot(critic_losses, color='orange', alpha=0.3, label="Raw Loss")
        axes[1].plot(smooth(critic_losses, weight=0.9), color='darkorange', linewidth=2, label="Smoothed")
        axes[1].set_title("Critic Loss (MSE) over Update Steps")
        axes[1].set_ylabel("L1 Loss")
        axes[1].set_xlabel("Update Step")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No Critic Loss Data', ha='center', va='center')

    # 3. Plot Actor Loss
    if actor_losses:
        axes[2].plot(actor_losses, color='green', alpha=0.3, label="Raw Loss")
        axes[2].plot(smooth(actor_losses, weight=0.9), color='darkgreen', linewidth=2, label="Smoothed")
        axes[2].set_title("Actor Total Loss over Update Steps (Minimizing -Q + MSE)")
        axes[2].set_ylabel("Loss")
        axes[2].set_xlabel("Update Step")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'No Actor Loss Data', ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle
    
    # Save the figure
    save_path = "training_metrics.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to '{save_path}'")
    
    # Show the interactive plot
    plt.show()

if __name__ == "__main__":
    parse_and_plot()