import os
import sys

# Setup SUMO paths programmatically before other imports
from scale_network import setup_sumo_env
setup_sumo_env()

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import sumo_rl
from env_setup import QueueObservationFunction, custom_reward_fn
from marl_algorithms import (
    FixedTimeController, TabularQLearningAgent, SARSAAgent,
    WLearningAgent, JALAgent, MaxPlusAgent, run_max_plus_coordination,
    QMIXAgentNetwork
)

def adapt_obs_dict(obs_dict):
    """Adapts all observations in the dictionary to have exactly size 2."""
    adapted = {}
    for agent_id, obs in obs_dict.items():
        obs = np.array(obs, dtype=np.float32)
        if len(obs) > 2:
            adapted[agent_id] = obs[:2]
        elif len(obs) < 2:
            adapted[agent_id] = np.pad(obs, (0, 2 - len(obs)), mode="constant")
        else:
            adapted[agent_id] = obs
    return adapted

def parse_args():
    parser = argparse.ArgumentParser(description="Watch pre-trained MARL agents in SUMO-GUI.")
    parser.add_argument("--size", type=int, default=2, choices=[2, 3, 4, 5], help="Grid network size.")
    parser.add_argument("--algo", type=str, default="fixed",
                        choices=["fixed", "iql_tabular", "sarsa_tabular", "w_learning", "jal", "max_plus", "qmix", "iql_deep"],
                        help="Algorithm to watch in GUI.")
    parser.add_argument("--delay", type=float, default=0.1, help="Simulation step sleep delay in seconds.")
    return parser.parse_args()


def load_policy(algo, agent_ids, size):
    """Loads a pre-trained policy for each agent from disk."""
    agents = {}
    
    # Grid size specific naming
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    for agent_id in agent_ids:
        if algo == "fixed":
            agents[agent_id] = FixedTimeController(agent_id)
        elif algo == "iql_tabular":
            agent = TabularQLearningAgent(agent_id)
            path = os.path.join(model_dir, f"{algo}_{agent_id}_{size}x{size}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded tabular Q-table for agent {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with un-trained/empty policy.")
            agents[agent_id] = agent
        elif algo == "sarsa_tabular":
            agent = SARSAAgent(agent_id)
            path = os.path.join(model_dir, f"{algo}_{agent_id}_{size}x{size}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded SARSA Q-table for agent {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with empty policy.")
            agents[agent_id] = agent
        elif algo == "w_learning":
            agent = WLearningAgent(agent_id)
            # Load Sub-Q tables and W tables
            for i in range(2):
                q_path = os.path.join(model_dir, f"w_learning_q_{i}_{agent_id}_{size}x{size}.npy")
                w_path = os.path.join(model_dir, f"w_learning_w_{i}_{agent_id}_{size}x{size}.npy")
                if os.path.exists(q_path):
                    agent.q_tables[i] = np.load(q_path)
                if os.path.exists(w_path):
                    agent.w_tables[i] = np.load(w_path)
            print(f"Loaded W-Learning tables for agent {agent_id}")
            agents[agent_id] = agent
        elif algo == "jal":
            # For JAL, count physical neighbors in KxK grid
            col = ord(agent_id[0]) - 65
            row = int(agent_id[1])
            neighbor_ids = []
            if col > 0: neighbor_ids.append(f"{chr(65 + col - 1)}{row}")
            if col < size - 1: neighbor_ids.append(f"{chr(65 + col + 1)}{row}")
            if row > 0: neighbor_ids.append(f"{chr(65 + col)}{row - 1}")
            if row < size - 1: neighbor_ids.append(f"{chr(65 + col)}{row + 1}")
            
            agent = JALAgent(agent_id, neighbor_ids)
            path = os.path.join(model_dir, f"jal_{agent_id}_{size}x{size}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded JAL Q-table for agent {agent_id}")
            agents[agent_id] = agent
        elif algo == "max_plus":
            col = ord(agent_id[0]) - 65
            row = int(agent_id[1])
            neighbor_ids = []
            if col > 0: neighbor_ids.append(f"{chr(65 + col - 1)}{row}")
            if col < size - 1: neighbor_ids.append(f"{chr(65 + col + 1)}{row}")
            if row > 0: neighbor_ids.append(f"{chr(65 + col)}{row - 1}")
            if row < size - 1: neighbor_ids.append(f"{chr(65 + col)}{row + 1}")
            
            agent = MaxPlusAgent(agent_id, neighbor_ids)
            # Load local and pair Q tables
            local_path = os.path.join(model_dir, f"maxplus_local_{agent_id}_{size}x{size}.npy")
            if os.path.exists(local_path):
                agent.q_local = np.load(local_path)
            for n_id in neighbor_ids:
                pair_path = os.path.join(model_dir, f"maxplus_pair_{agent_id}_{n_id}_{size}x{size}.npy")
                if os.path.exists(pair_path):
                    agent.q_pairs[n_id] = np.load(pair_path)
            print(f"Loaded Max-Plus matrices for agent {agent_id}")
            agents[agent_id] = agent
        elif algo == "qmix":
            agent = QMIXAgentNetwork(obs_dim=2, action_dim=2)
            path = os.path.join(model_dir, f"qmix_{agent_id}_{size}x{size}.pth")
            if os.path.exists(path):
                agent.load_state_dict(torch.load(path))
                agent.eval()
                print(f"Loaded QMIX agent network for {agent_id} from {path}")
            agents[agent_id] = agent
        elif algo == "iql_deep":
            # Backward compatibility with trained PyTorch model
            from train import QNet
            agent = QNet(state_shape=2, action_shape=2)
            path = os.path.join("models", "iql_policy.pth")
            if os.path.exists(path):
                state_dict = torch.load(path, map_location="cpu")
                # Strip Tianshou policy prefixes if present
                clean_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        clean_dict[k.replace("model.", "")] = v
                    else:
                        clean_dict[k] = v
                agent.load_state_dict(clean_dict, strict=False)
                agent.eval()
                print(f"Loaded Deep IQL network from {path}")
            agents[agent_id] = agent
            
    return agents

def run_gui_simulation():
    args = parse_args()
    net_file = f"grid_{args.size}x{args.size}.net.xml"
    rou_file = f"grid_{args.size}x{args.size}.rou.xml"
    
    if not os.path.exists(net_file) or not os.path.exists(rou_file):
        print(f"Error: Network files for {args.size}x{args.size} grid do not exist.")
        print("Please run 'scale_network.py --size [N]' to generate them first.")
        sys.exit(1)
        
    print(f"\nLaunching SUMO-GUI for algorithm '{args.algo}' on network size {args.size}x{args.size}...")
    
    # Initialize parallel environment directly (forces SUMO-GUI)
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=rou_file,
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        reward_fn=custom_reward_fn,
        observation_class=QueueObservationFunction
    )
    
    agent_ids = env.possible_agents
    agents = load_policy(args.algo, agent_ids, args.size)
    
    # Step through one episode slowly
    step = 0
    done = False
    
    # Dictionaries to maintain step action histories for JAL / Max-Plus
    obs_dict = {a_id: np.zeros(2) for a_id in agent_ids}
    action_dict = {a_id: 0 for a_id in agent_ids}
    
    print("\nStarting Real-time Decision Monitoring...")
    print("-" * 100)
    
    # Reset parallel env
    obs_dict, infos = env.reset()
    obs_dict = adapt_obs_dict(obs_dict)
    
    try:
        while env.agents:
            step += 5  # sumo-rl environment delta_time is 5 seconds
            
            # 1. Action selection
            if args.algo == "max_plus":
                # Joint message passing coordination
                actions = run_max_plus_coordination(agents, obs_dict)
            else:
                actions = {}
                for agent_id in env.agents:
                    obs = obs_dict[agent_id]
                    agent = agents[agent_id]
                    
                    if args.algo in ["qmix", "iql_deep"]:
                        # PyTorch forward pass
                        with torch.no_grad():
                            obs_t = torch.FloatTensor(obs).unsqueeze(0)
                            out = agent(obs_t)
                            # Handle QNet returning (q, state) tuple vs QMIXAgentNetwork returning tensor
                            q_vals = out[0] if isinstance(out, tuple) else out
                            actions[agent_id] = int(q_vals.argmax(dim=-1).item())
                    elif args.algo == "jal":
                        # Feed adjacent neighbors' previous action context
                        actions[agent_id] = agent.compute_action(obs, explore=False)
                    else:
                        # Classical tabular compute_action
                        actions[agent_id] = agent.compute_action(obs, explore=False)

            # Ensure all selected actions are mathematically safe for each agent's active action space
            for agent_id in list(actions.keys()):
                action_space = env.action_space(agent_id)
                if hasattr(action_space, "n"):
                    actions[agent_id] = int(actions[agent_id] % action_space.n)
            
            # Print decision logging to console in real-time in requested format
            for agent_id in env.agents:
                obs = obs_dict[agent_id]
                act = actions.get(agent_id, 0)
                phase_name = "Green North-South" if act == 0 else "Green East-West"
                print(f"Step {step:04d} | Agent '{agent_id}' | State (Queues) - N: {int(obs[0])}, S: 0, E: {int(obs[1])}, W: 0 | Action Selected: {phase_name}")
                
            # Step the environment
            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            next_obs_dict = adapt_obs_dict(next_obs_dict)
            
            # Store history
            obs_dict = next_obs_dict
            action_dict = actions
            
            # Small delay for visual inspection
            time.sleep(args.delay)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        env.close()
        print("-" * 100)
        print("SUMO-GUI session finished successfully.")

if __name__ == "__main__":
    run_gui_simulation()
