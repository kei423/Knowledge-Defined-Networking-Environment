import networkx as nx
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

class RoutingEnv(ParallelEnv):
    def __init__(self, nodes=14, drop_prob=0.1):
        # Using a standard topology like NSFNET (14 nodes)
        self.base_graph = nx.erdos_renyi_graph(n=nodes, p=0.3, seed=42)
        # Doing only a single agent serving as the central controller
        self.agents = ["controller"]
        self.drop_prob = drop_prob
        self.active_nodes = list(self.base_graph.nodes)
        
        # Observation: Adjacency matrix + Link loads
        # shape = [# nodes, # nodes]
        self.observation_spaces = {
            "controller": spaces.Box(low=0, high=1, shape=(nodes, nodes), dtype=np.float32)
        }
        # Action: A vector of link weights (continuous)
        # Each edge has some positive weight that represents its cost
        # Each weight has to be non-negative to work with the Dijkstra-based OSPF algo
        self.action_spaces = {
            "controller": spaces.Box(low=0.1, high=10.0, shape=(len(self.base_graph.edges),), dtype=np.float32)
        }

    def reset(self, seed=None):
        self.active_nodes = list(self.base_graph.nodes)
        # Randomly drop nodes to simulate down stations in realw-world traffic
        for node in self.base_graph.nodes:
            if np.random.rand() < self.drop_prob:
                self.active_nodes.remove(node)
        
        # Construct the current view of the network
        current_adj = nx.to_numpy_array(self.base_graph.subgraph(self.active_nodes))
        return {"controller": current_adj}, {}

    def step(self, actions):
        pass