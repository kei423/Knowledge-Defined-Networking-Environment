from routingenv import RoutingEnv
import numpy as np

def ospf_action(env: RoutingEnv) -> np.ndarray:
    # assume a basic OSPF where all link weights are the same and is merely
    # routing based on number of hops
    return np.ones(env.n_edges, dtype=np.float32)

env = RoutingEnv(nodes=12, drop_prob=0.10, max_steps=200, traffic_scale=0.9)

obs, _ = env.reset(seed=0)
ospf_rewards = []
while env.agents:
    action = ospf_action(env)
    result = env.step({"controller": action})
    obs, rewards, terms, truncs, infos = result
    ospf_rewards.append(rewards["controller"])

print(f"OSPF avg reward: {np.mean(ospf_rewards):.4f}")
