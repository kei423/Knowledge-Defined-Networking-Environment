import torch
import torch.nn as nn
import numpy as np
import random
import logging
import sys
from collections import deque
from routingenv import RoutingEnv
from mpdrl_test import MPDRLActor, MPDRLCritic, OUNoise

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
# Set up logging to write DEBUG and above to a file, and INFO and above to console
log_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s')

file_handler = logging.FileHandler("mpdrl_simulation.log", mode='w')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger("Runner")

# Mute the verbose per-forward-pass logs from the neural network module to avoid log flooding
logging.getLogger("MPDRL_Net").setLevel(logging.INFO)

# ---------------------------------------------------------
# Hyperparameters (Matched to paper specifications)
# ---------------------------------------------------------
BATCH_SIZE = 16
GAMMA = 0.99
TAU = 0.01          # Soft update parameter
LR_ACTOR = 1e-5     # alpha / beta from paper
LR_CRITIC = 1e-6
EPISODES = 20       # Number of training episodes
MAX_STEPS = 200     # Steps per episode
BUFFER_CAPACITY = 10000

# ---------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# Main Training Loop (Algorithm 2)
# ---------------------------------------------------------
def train_mpdrl():
    logger.info("Initializing MPDRL Training Environment and Buffers...")
    
    # 3: Initialize environment env and replay buffer B
    env = RoutingEnv(nodes=14, drop_prob=0.10, max_steps=MAX_STEPS)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    noise = OUNoise(size=env.n_edges)
    
    obs_dict, _ = env.reset(seed=42)
    obs = obs_dict["controller"]
    flat_obs_dim = obs["edge_features"].flatten().shape[0]
    
    logger.info(f"Observation flattened dimension: {flat_obs_dim}, Action dimension (edges): {env.n_edges}")
    
    # 1: Initialize actor network and critic network with random weights
    actor = MPDRLActor(num_edges=env.n_edges)
    critic = MPDRLCritic(obs_flat_dim=flat_obs_dim, action_dim=env.n_edges)
    
    # 2: Initialize target networks with weights
    target_actor = MPDRLActor(num_edges=env.n_edges)
    target_critic = MPDRLCritic(obs_flat_dim=flat_obs_dim, action_dim=env.n_edges)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    logger.info("Starting MPDRL Training Loop...")

    # 4: for each episode do
    for episode in range(EPISODES):
        logger.info(f"--- Starting Episode {episode + 1}/{EPISODES} ---")
        
        # 5: reset environment
        obs_dict, _ = env.reset()
        obs = obs_dict["controller"]
        noise.reset()
        episode_reward = 0
        
        # The edge_index is static across the simulation (link drops only zero-out features)
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        
        # 6: for step t = 1 to step_num do
        while env.agents:
            step_idx = env._step
            logger.debug(f"[Ep {episode+1} | Step {step_idx}] Processing step...")
            
            edge_features = torch.tensor(obs["edge_features"], dtype=torch.float32)
            
            # 7: Select action a_t = mu(s_t) + phi_t (Actor + Noise)
            logger.debug("Starting forward pass for action selection...")
            with torch.no_grad():
                action_tensor = actor(edge_index, edge_features)
            logger.debug("Finished forward pass for action selection.")
            
            # Convert safely to avoid Numpy 2.x issue
            base_action = np.array(action_tensor.tolist())
            action_noise = noise.sample()
            action = np.clip(base_action + action_noise, 0.1, 10.0)
            
            logger.debug(f"Action Selected. Base mean: {base_action.mean():.4f}, Noise mean: {action_noise.mean():.4f}")
            
            # 8: Execute action, observe reward and next state
            result = env.step({"controller": action})
            next_obs_dict, rewards, terms, truncs, infos = result
            next_obs = next_obs_dict["controller"]
            reward = rewards["controller"]
            done = terms["controller"]
            
            next_edge_features = torch.tensor(next_obs["edge_features"], dtype=torch.float32)
            
            # 9: Store the transition sample in B
            buffer.push(edge_features, action, reward, next_edge_features, done)
            logger.debug(f"Transition pushed to buffer. Buffer size: {len(buffer)}")
            
            obs = next_obs
            episode_reward += reward
            
            # 10: Sample a random minibatch of N samples from B
            if len(buffer) >= BATCH_SIZE:
                logger.debug(f"Triggering DDPG update. Sampling batch of size {BATCH_SIZE}...")
                batch = buffer.sample(BATCH_SIZE)
                
                # Unpack batch and structure dimensions [BATCH_SIZE, ...]
                state_batch = torch.stack([b[0] for b in batch])
                action_batch = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
                reward_batch = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)
                next_state_batch = torch.stack([b[3] for b in batch])
                done_batch = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32).unsqueeze(1)
                
                # ----------------------------------------------------
                # 11 & 13: Compute L1 Loss and Update Critic Network
                # ----------------------------------------------------
                logger.debug("Starting batch forward passes for Target Actor and Critic...")
                with torch.no_grad():
                    # Evaluate target actor (using list comp inside torch.stack to aggregate)
                    next_actions = torch.stack([target_actor(edge_index, next_state_batch[i]) for i in range(BATCH_SIZE)])
                    # Vectorized Critic forward
                    Q_targets_next = target_critic(next_state_batch.flatten(1), next_actions) # [BATCH_SIZE, 1]
                logger.debug("Finished batch forward passes for Target networks.")
                
                # y = r + gamma * Q'(s_{t+1}, mu'(s_{t+1}))
                y_expected = reward_batch + (1 - done_batch) * GAMMA * Q_targets_next
                
                # Vectorized Critic forward for current state
                Q_expected = critic(state_batch.flatten(1), action_batch) # [BATCH_SIZE, 1]
                
                # Critic MSE Loss
                critic_loss = nn.MSELoss()(Q_expected, y_expected.detach())
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_optimizer.step()
                
                logger.debug(f"Critic Update: L1 (MSE) Loss = {critic_loss.item():.6f}")
                
                # ----------------------------------------------------
                # 11 & 14: Compute L2/J Loss and Update Actor Network
                # ----------------------------------------------------
                logger.debug("Starting batch forward passes for Current Actor...")
                curr_actions = torch.stack([actor(edge_index, state_batch[i]) for i in range(BATCH_SIZE)])
                logger.debug("Finished batch forward passes for Current Actor.")
                
                # J Loss: Vectorized calculation 
                actor_loss_J = critic(state_batch.flatten(1), curr_actions).mean()
                
                # L2 Loss (Minimize MSE against reward + gamma * next_action)
                # PERFORMANCE WIN: Re-using `next_actions` from the target_actor we computed above
                target_L2 = reward_batch + GAMMA * next_actions 
                actor_loss_L2 = nn.MSELoss()(curr_actions, target_L2.detach())
                
                # Total Actor Loss (-J + L2)
                actor_loss = -actor_loss_J + actor_loss_L2
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                logger.debug(f"Actor Update: J_Loss = {-actor_loss_J.item():.6f}, L2_Loss = {actor_loss_L2.item():.6f}, Total_Loss = {actor_loss.item():.6f}")
                
                # ----------------------------------------------------
                # 15: Soft Update Target Networks
                # ----------------------------------------------------
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                    
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                
                logger.debug("Soft update applied to Target Actor and Critic networks.")
                    
        logger.info(f"Episode {episode + 1}/{EPISODES} Completed | Avg Step Reward (-LU_SD): {episode_reward/MAX_STEPS:.4f}")

if __name__ == "__main__":
    train_mpdrl()