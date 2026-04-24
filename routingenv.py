import networkx as nx
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

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
        self.n_edges = len(self.edge_list)

        # Doing only a single agent serving as the central controller
        self.agents = ["controller"]
        self.possible_agents = ["controller"]
        
        # Observation: Adjacency matrix + Link loads
        # both shapes = shape = [# nodes, # nodes] = n x n
        # flattened together = n x n x 2
        obs_size = nodes * nodes * 2
        self.observation_spaces = {
            "controller": spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        }
        # Action: A vector of link weights representing the edges (continuous)
        # Each edge has some positive weight that represents its cost
        # Each weight has to be non-negative to work with the Dijkstra-based OSPF algo
        self.action_spaces = {
            "controller": spaces.Box(low=0.1, high=10.0, shape=(self.n_edges,), dtype=np.float32)
        }

        self._step = 0
        self.current_graph = None
        self.link_loads = None # n x n matrrix of link utilization [0, 1]
        self.traffic_matrix = None # n x n matrix of demands


    def _generate_traffic_matrix(self):
        # returns a matrix of random traffic demand between all node pairs
        traffic_matrix = np.random.exponential(scale=self.traffic_scale, size=(self.n, self.n))
        np.fill_diagonal(traffic_matrix, 0) # no traffic from a node to itself
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
        return g
    
    def _compute_link_loads(self, graph, weights: dict, traffic_matrix):
        # route all edge src and dst with weighted Dijkstra
        # compute total flow on each link and then normalize to capacity (uniform = 1)
        # returns n x n load matrix
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
                    # demand is dropped — counts as loss, no load added
                    continue
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    load[u, v] += demand
                    load[v, u] += demand  # undirected

        # normalize --> clip to [0, 1] (values > 1 = congested)
        # max_load = load.max()
        # if max_load > 0:
        #     load = np.clip(load / max_load, 0, 1)
        if total_demand > 0:
            load = load / total_demand
        return load
    
    def _set_edge_weights(self, graph, action: np.ndarray):
        # applies agent's weight vectors to current graph edges
        for idx, (u, v) in enumerate(self.edge_list):
            if graph.has_edge(u, v):
                graph[u][v]["weight"] = float(action[idx])
    
    def _get_obs(self, graph, link_loads):
        # returns observation (adjacency matrix + link loads)
        adjacency_matrix = nx.to_numpy_array(graph, nodelist=range(self.n), dtype=np.float32)
        obs = np.concatenate([adjacency_matrix.flatten(), link_loads.flatten()])
        return obs.astype(np.float32)

    def _compute_reward(self, graph, link_loads, traffic_matrix):
        # returns reward
        # penalizes max link utilization (congested traffic)
        # penalizes dropped flows (unreachable src-dst pairs)

        # max_util = worst case link load for current network
        max_util = link_loads.max()

        # dropped penalty
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

        reward = -(max_util + drop_rate)  # both in [0,1], so reward in [-2, 0]
        return float(reward)

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
        self.link_loads = self._compute_link_loads(self.current_graph,
                                                   {},
                                                   self.traffic_matrix)
        
        # compute reward
        reward = self._compute_reward(self.current_graph,
                                      self.link_loads,
                                      self.traffic_matrix)
        
        # apply new link failures and traffic for next step
        self._step += 1
        terminated = self._step >= self.max_steps

        # update env for next observation
        self.current_graph = self._apply_link_failures()
        self.traffic_matrix = self._generate_traffic_matrix()
        self.link_loads = np.zeros((self.n, self.n), dtype=np.float32)

        obs = self._get_obs(self.current_graph, self.link_loads)

        rewards = {"controller": reward}
        terminations = {"controller": terminated}
        truncations = {"controller": False}
        infos = {"controller": {"max_util": self.link_loads.max(), "step": self._step}}

        if terminated:
            self.agents = []

        return (
            {"controller": obs},
            rewards,
            terminations,
            truncations,
            infos,
        )
