# Message Passing Deep Reinforcement Learning Implementation for Routing Optimization

Based on the design proposed in IEEE 10.1109/TMC.2023.3235446. 

This project aims to implement an actor-critic model for routing optimization on dynamic networks, with a focus on adapting to transient failures. 

## Repo Structure

  - `mpdrl_test.py`: Actor-critic network implementations. Graph Neural Network with LRU for Actor. Standard MLP for Critic. 
  - `ospf_test.py`: Baseline OSPF implementation.
  - `plot_logs.py`: Generate graphs to visualize training progression.
  - `routingenv.py`: PettingZoo environment for dynamic network simulation. 
  - `runner.py`: Implements the paper's DDPG training loop and executes simulations. 
  - `requirements.txt`: Names dependencies
  - `mpdrl_simulation.log`: Training and simulation log file

## Instructions to run

- Ensure python3.10 or newer is installed
- Clone repo
- run `pip install -r requirements.txt` to install all dependencies
- run `python3 runner.py` to execute the training and evaluation loop. All relevant outputs will be stored `mpdrl_simulation.log`. 
- run `plot_logs.py` to get an overview of the agent's training progression
