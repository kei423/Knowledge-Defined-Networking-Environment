import networkx as nx
import numpy as np
import logging
from gymnasium import spaces
from pettingzoo import ParallelEnv

logger = logging.getLogger("RoutingEnv")

class RoutingEnv(ParallelEnv):
    metadata = {"name": "routing_v0"}
    def __init__(self, nodes=14, drop_prob=0.1, max_steps=100, traffic_scale=1.0):
        # nodes = number of nodes in the network
        self.n = nodes
        # drop_prob = drop probability of edges to simulate link failure
        self.drop_prob = drop_prob
        # max_steps = max steps ran for the simulation
        self.max_steps = max_steps
        # traffic_scale = to simulate how congested the traffic is
        self.traffic_scale = traffic_scale
    
        # generate a fixed standard topology like NSFNET (14 nodes)
        self.base_graph = nx.erdos_renyi_graph(n=nodes, p=0.3, seed=42)

        self.edge_list = list(self.base_graph.edges())  # fixed order
        # Make it a directed representation for message passing (u->v and v->u)
        self.directed_edge_list = self.edge_list + [(v, u) for (u, v) in self.edge_list]
        self.n_edges = len(self.edge_list)
        self.n_directed_edges = len(self.directed_edge_list)
        
        logger.debug(f"Initialized RoutingEnv: Nodes={self.n}, Undirected Edges={self.n_edges}, Directed Edges={self.n_directed_edges}")

        # Doing only a single agent serving as the central controller
        self.agents = ["controller"]
        self.possible_agents = ["controller"]
        
        # Observation: Graph structured data (edge_index and edge_features)
        # edge_index: [2, n_directed_edges]
        # edge_features: [n_directed_edges, 20] (Capacity, Utilization, Weight, 17x zeros)
        self.observation_spaces = {
            "controller": spaces.Dict({
                "edge_index": spaces.Box(low=0, high=self.n-1, shape=(2, self.n_directed_edges), dtype=np.int64),
                "edge_features": spaces.Box(low=0, high=np.inf, shape=(self.n_directed_edges, 20), dtype=np.float32)
            })
        }
        
        # Action: A vector of link weights representing the edges (continuous)
        self.action_spaces = {
            "controller": spaces.Box(low=0.1, high=10.0, shape=(self.n_edges,), dtype=np.float32)
        }

        self._step = 0
        self.current_graph = None
        self.link_loads = None # n x n matrix of link utilization
        self.traffic_matrix = None # n x n matrix of demands


    def _generate_traffic_matrix(self):
        # returns a matrix of random traffic demand between all node pairs
        traffic_matrix = np.random.exponential(scale=self.traffic_scale, size=(self.n, self.n))
        np.fill_diagonal(traffic_matrix, 0) # no traffic from a node to itself
        
        total_demand = traffic_matrix.sum()
        logger.debug(f"Generated traffic matrix. Total demand: {total_demand:.2f}")
        return traffic_matrix
    
    def _apply_link_failures(self):
        # returns a subgraph with edges randomly dropped
        g = self.base_graph.copy()
        failed = []
        for u, v in list(g.edges()):
            if np.random.rand() < self.drop_prob:
                g.remove_edge(u, v)
                failed.append((u, v))
                
        # guarantees at least a spanning tree exists or the network wouldn't function
        if not nx.is_connected(g):
            # re-add minimum edges to reconnect
            for u, v in failed:
                if not nx.is_connected(g):
                    g.add_edge(u, v)
                    failed.remove((u,v))
                    
        logger.debug(f"Link failures applied. {len(failed)} links effectively dropped out of {self.n_edges}.")
                    
        # Initialize default weights
        for u, v in g.edges():
            g[u][v]['weight'] = 1.0
            
        return g
    
    def _compute_link_loads(self, graph, traffic_matrix):
        # route all edge src and dst with weighted Dijkstra
        load = np.zeros((self.n, self.n), dtype=np.float32)
        nodes = list(graph.nodes())
        total_demand = traffic_matrix.sum()

        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue
                demand = traffic_matrix[src, dst]
                if demand == 0:
                    continue
                try:
                    path = nx.dijkstra_path(graph, src, dst, weight="weight")
                except nx.NetworkXNoPath:
                    # demand is dropped
                    continue
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    load[u, v] += demand
                    load[v, u] += demand  # undirected

        # normalize against total demand
        if total_demand > 0:
            load = load / total_demand
            
        logger.debug(f"Link loads computed. Max normalized edge load: {load.max():.4f}")
        return load
    
    def _set_edge_weights(self, graph, action: np.ndarray):
        # applies agent's weight vectors to current graph edges
        for idx, (u, v) in enumerate(self.edge_list):
            if graph.has_edge(u, v):
                # Ensure weight is strictly positive for Dijkstra
                graph[u][v]["weight"] = max(float(action[idx]), 0.01)
    
    def _get_obs(self, graph, link_loads):
        # Construct PyTorch Geometric compatible edge_index and features
        edge_index = []
        edge_features = []
        
        for u, v in self.directed_edge_list:
            edge_index.append([u, v])
            
            # If edge exists in current (potentially failing) graph
            if graph.has_edge(u, v):
                capacity = 1.0  # Normalized capacity
                utilization = link_loads[u, v]
                weight = graph[u][v].get("weight", 1.0)
            else:
                # Failed link
                capacity = 0.0
                utilization = 0.0
                weight = 0.0
                
            # [x1, x2, x3, 0, ..., 0] -> Size 20
            feat = [capacity, utilization, weight] + [0.0] * 17
            edge_features.append(feat)
            
        edge_index = np.array(edge_index, dtype=np.int64).T # Shape: [2, E]
        edge_features = np.array(edge_features, dtype=np.float32) # Shape: [E, 20]
        
        return {
            "edge_index": edge_index,
            "edge_features": edge_features
        }

    def _compute_reward(self, graph, link_loads, traffic_matrix):
        # MPDRL Objective: Minimize Standard Deviation of Link Utilization
        
        active_loads = []
        for u, v in graph.edges():
            active_loads.append(link_loads[u, v])
            
        if len(active_loads) > 0:
            lu_sd = np.std(active_loads)
        else:
            lu_sd = 0.0

        # Calculate dropped flows to act as a harsh penalty constraint
        dropped = 0
        total = 0
        nodes = list(graph.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst or traffic_matrix[src, dst] == 0:
                    continue
                total += 1
                if not nx.has_path(graph, src, dst):
                    dropped += 1

        drop_rate = dropped / total if total > 0 else 0

        # Reward is negative Standard Deviation of Link Utilization
        reward = -(lu_sd + drop_rate)  
        return float(reward), float(lu_sd), float(drop_rate)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self._step = 0
        self.agents = ["controller"]
        self.current_graph = self._apply_link_failures()
        self.traffic_matrix = self._generate_traffic_matrix()

        # Initialize loads to zero (no routing decision yet)
        self.link_loads = np.zeros((self.n, self.n), dtype=np.float32)

        obs = self._get_obs(self.current_graph, self.link_loads)
        logger.debug("Environment Reset Complete.")
        return {"controller": obs}, {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions):
        action = actions["controller"]

        # apply weights to current graph
        self._set_edge_weights(self.current_graph, action)

        # simulate traffic routing and compute link load
        self.link_loads = self._compute_link_loads(self.current_graph, self.traffic_matrix)
        
        # compute reward
        reward, lu_sd, drop_rate = self._compute_reward(self.current_graph, self.link_loads, self.traffic_matrix)
        logger.debug(f"Step {self._step} metrics -> Reward: {reward:.4f}, LU_SD: {lu_sd:.4f}, Drop Rate: {drop_rate:.4f}")
        
        # apply new link failures and traffic for next step
        self._step += 1
        terminated = self._step >= self.max_steps

        # update env for next observation
        self.current_graph = self._apply_link_failures()
        self.traffic_matrix = self._generate_traffic_matrix()
        
        # Reset loads for next state observation calculation
        self.link_loads = np.zeros((self.n, self.n), dtype=np.float32)
        obs = self._get_obs(self.current_graph, self.link_loads)

        rewards = {"controller": reward}
        terminations = {"controller": terminated}
        truncations = {"controller": False}
        infos = {"controller": {"lu_sd": lu_sd, "drop_rate": drop_rate, "step": self._step}}

        if terminated:
            logger.debug("Max steps reached. Terminating environment episode.")
            self.agents = []

        return (
            {"controller": obs},
            rewards,
            terminations,
            truncations,
            infos,
        )