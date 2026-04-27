import torch
import torch.nn as nn
import numpy as np
import logging
from routingenv import RoutingEnv

logger = logging.getLogger("MPDRL_Net")

# ---------------------------------------------------------
# 1. MPDRL Neural Network Architectures
# ---------------------------------------------------------

class MPDRLActor(nn.Module):
    def __init__(self, num_edges, edge_features=20):
        super().__init__()
        self.num_edges = num_edges
        self.edge_features_dim = edge_features
        
        # Cache for static graph adjacency indices
        self.adj_src = None
        self.adj_nbr = None

        self.message_fn = nn.Sequential(
            nn.Linear(edge_features * 2, 32),  # concat source + neighbor hidden states
            nn.SELU(),
            nn.Linear(32, edge_features)
        )
        self.update_fn = nn.GRUCell(input_size=edge_features, hidden_size=edge_features)
        self.readout_fn = nn.Sequential(
            nn.Linear(edge_features, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def _build_edge_adj(self, edge_index, num_directed_edges):
        """
        For directed edges, edge i=(u→v) is adjacent to edge j=(v→w).
        Returns: (source_edge_idx, neighbor_edge_idx) for scatter aggregation.
        """
        src_nodes = edge_index[0]  # shape [E]
        dst_nodes = edge_index[1]  # shape [E]

        adj_src, adj_nbr = [], []
        # Build a lookup: node -> list of outgoing edge indices
        node_to_out_edges = {}
        for j in range(num_directed_edges):
            # We want the source node of edge j (where the edge leaves from)
            u = src_nodes[j].item()  
            node_to_out_edges.setdefault(u, []).append(j)

        for i in range(num_directed_edges):
            v = dst_nodes[i].item()  # head of edge i
            for j in node_to_out_edges.get(v, []):
                if i != j:
                    adj_src.append(i)   # edge i aggregates from neighbor j
                    adj_nbr.append(j)

        return (torch.tensor(adj_src, dtype=torch.long),
                torch.tensor(adj_nbr, dtype=torch.long))

    def forward(self, edge_index, edge_attr):
        E = edge_attr.shape[0]
        h_t = edge_attr.clone()
        
        logger.debug(f"Actor Forward Pass: Input shape {h_t.shape}")
        
        # build the adjacency graph once and cache it
        if self.adj_src is None or self.adj_nbr is None:
            self.adj_src, self.adj_nbr = self._build_edge_adj(edge_index, E)
            # Ensure indices are on the same device as the features
            self.adj_src = self.adj_src.to(edge_attr.device)
            self.adj_nbr = self.adj_nbr.to(edge_attr.device)
            logger.debug(f"Adjacency matrix built. Adjacency pairs: {self.adj_src.shape[0]}")

        for t in range(4):
            # Gather pairs: for each adjacency pair, concat h of edge i and neighbor j
            h_i = h_t[self.adj_src]   # [num_adj_pairs, F]
            h_j = h_t[self.adj_nbr]   # [num_adj_pairs, F]
            msgs = self.message_fn(torch.cat([h_i, h_j], dim=-1))  # [num_adj_pairs, F]

            # Aggregate (sum) messages into each edge
            agg = torch.zeros_like(h_t)
            agg.scatter_add_(0, self.adj_src.unsqueeze(1).expand_as(msgs), msgs)

            # GRU update: input=aggregated message, hidden=current state
            h_t = self.update_fn(agg, h_t)

        h_base = h_t[:self.num_edges]
        weights = self.readout_fn(h_base).squeeze(-1) + 0.1
        final_weights = torch.clamp(weights, min=0.1, max=10.0)
        
        logger.debug(f"Actor Forward Pass Complete: Output weights shape {final_weights.shape}")
        return final_weights

class MPDRLCritic(nn.Module):
    def __init__(self, obs_flat_dim, action_dim):
        super(MPDRLCritic, self).__init__()
        # 3-layer fully-connected feedforward neural network
        # 200 neurons in layer 1 and 2, ReLU activation, linear final layer
        self.net = nn.Sequential(
            nn.Linear(obs_flat_dim + action_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, state_flat, action):
        logger.debug(f"Critic Forward Pass: state_flat shape {state_flat.shape}, action shape {action.shape}")
        x = torch.cat([state_flat, action], dim=-1)
        q_value = self.net(x)
        logger.debug(f"Critic Forward Pass Complete: Q-value shape {q_value.shape}")
        return q_value

# ---------------------------------------------------------
# 2. Action Selection & Exploration Noise
# ---------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for action exploration in DDPG."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def mpdrl_action(env: RoutingEnv, actor: MPDRLActor, obs: dict, noise_process: OUNoise) -> np.ndarray:
    # Convert numpy observations to PyTorch tensors
    edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
    edge_features = torch.tensor(obs["edge_features"], dtype=torch.float32)
    
    # Get deterministic action from the Actor MPNN
    with torch.no_grad():
        action_tensor = actor(edge_index, edge_features)
        
    # Bypass PyTorch's internal NumPy bridge which fails on NumPy 2.x + PyTorch < 2.3
    # by converting the tensor to a standard Python list first
    action = np.array(action_tensor.tolist())
    
    # Equation 7: a_t = mu(s_t) + phi_t (Add OU Noise for exploration)
    noise = noise_process.sample()
    logger.debug(f"OUNoise sampled. Added to base action. Noise vector head: {noise[:3]}")
    action = action + noise
    
    # Ensure the final action remains within the valid environment bounds
    return np.clip(action, 0.1, 10.0)

# ---------------------------------------------------------
# 3. Main Simulation Loop
# ---------------------------------------------------------

def run_mpdrl_test():
    # Configure simple logger for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    
    # Initialize the updated environment
    env = RoutingEnv(nodes=12, drop_prob=0.30, max_steps=200)
    obs_dict, _ = env.reset(seed=0)
    obs = obs_dict["controller"]
    
    # Initialize Actor
    actor = MPDRLActor(num_edges=env.n_edges)
    
    # Note: Critic initialization requires flattened dimensions for standard DDPG.
    flat_obs_dim = obs["edge_features"].flatten().shape[0]
    critic = MPDRLCritic(obs_flat_dim=flat_obs_dim, action_dim=env.n_edges)
    
    # Initialize Noise Process
    ou_noise = OUNoise(size=env.n_edges)
    
    mpdrl_rewards = []
    
    logger.info("Running Untrained MPDRL Agent Simulation...")
    
    while env.agents:
        # Get action from the Actor network
        action = mpdrl_action(env, actor, obs, ou_noise)
        
        # Step through the environment
        result = env.step({"controller": action})
        obs_dict, rewards, terms, truncs, infos = result
        obs = obs_dict["controller"]
        
        mpdrl_rewards.append(rewards["controller"])

    mean_reward = np.mean(mpdrl_rewards)
    logger.info(f"MPDRL Average Reward (-LU_SD): {mean_reward:.4f}")
    logger.info("Note: This is an untrained agent exploring randomly with an MPNN.")

if __name__ == "__main__":
    run_mpdrl_test()