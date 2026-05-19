import os
import sys
import time
import argparse
import numpy as np
import torch

from simulator.env_setup import make_wave_env
from marl_algorithms import TabularQLearningAgent, QMIXAgentNetwork

class FixedTimeController:
    """Cycle traffic signals in a standard round-robin fashion."""
    def __init__(self, agent_id, cycle_steps=6):
        self.agent_id = agent_id
        self.cycle_steps = cycle_steps
        self.step_count = 0
        self.current_action = 0

    def compute_action(self, obs, explore=False):
        if self.step_count > 0 and self.step_count % self.cycle_steps == 0:
            self.current_action = 1 - self.current_action
        self.step_count += 1
        return self.current_action

def parse_args():
    parser = argparse.ArgumentParser(description="Watch pre-trained MARL agents in SUMO-GUI.")
    parser.add_argument("--algo", type=str, default="iql_tabular",
                        choices=["iql_tabular", "qmix", "iql_deep", "fixed"],
                        help="Algorithm to watch in GUI.")
    parser.add_argument("--delay", type=float, default=0.1, help="Simulation step sleep delay in seconds.")
    return parser.parse_args()

def load_policy(algo, agent_ids):
    """Loads a pre-trained policy for each agent from disk."""
    agents = {}
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    for agent_id in agent_ids:
        if algo == "iql_tabular":
            agent = TabularQLearningAgent(agent_id, num_states=625)
            # Adjust model path based on naming convention
            path = os.path.join(model_dir, f"{algo}_{agent_id}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded tabular Q-table for agent {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with un-trained/empty policy.")
            agents[agent_id] = agent
            
        elif algo == "qmix":
            agent = QMIXAgentNetwork(obs_dim=2, action_dim=2)
            path = os.path.join(model_dir, f"qmix_{agent_id}.pth")
            if os.path.exists(path):
                agent.load_state_dict(torch.load(path))
                agent.eval()
                print(f"Loaded QMIX agent network for {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with un-trained/empty policy.")
            agents[agent_id] = agent
            
        elif algo == "iql_deep":
            try:
                from train import QNet
                agent = QNet(state_shape=2, action_shape=2)
                path = os.path.join("models", "iql_policy.pth")
                if os.path.exists(path):
                    state_dict = torch.load(path, map_location="cpu")
                    # Strip Tianshou policy prefixes if present
                    clean_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                    agent.load_state_dict(clean_dict, strict=False)
                    agent.eval()
                    print(f"Loaded Deep IQL network from {path}")
                agents[agent_id] = agent
            except ImportError:
                print("Warning: Could not import QNet from train.py.")
                
        elif algo == "fixed":
            agent = FixedTimeController(agent_id)
            print(f"Loaded Fixed-Time Controller for agent {agent_id}")
            agents[agent_id] = agent
                
    return agents

def run_gui_simulation():
    args = parse_args()
    
    print(f"\nLaunching SUMO-GUI for algorithm '{args.algo}' on the 1x4 Green Wave Scenario...")
    
    # Generate the wave network if it doesn't exist
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    if not os.path.exists(net_file) or not os.path.exists(rou_file):
        from simulator.generate_1x4_wave import build_1x4_scenario
        build_1x4_scenario()
        
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=3600, delta_time=5, use_gui=True)
    env.reset()
    
    agent_ids = env.possible_agents
    agents = load_policy(args.algo, agent_ids)
    
    print("\nStarting Real-time Decision Monitoring...")
    print("-" * 100)
    
    step = 0
    
    try:
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                agent = agents.get(agent_id)
                obs = np.array(observation, dtype=np.float32)
                
                if args.algo in ["qmix", "iql_deep"]:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        out = agent(obs_t)
                        q_vals = out[0] if isinstance(out, tuple) else out
                        action = int(q_vals.argmax(dim=-1).item())
                else:
                    action = agent.compute_action(obs, explore=False)
                    
                # Action mapping (AEC loops handle one agent per time step, delay after all actions are processed could be added)
                phase_name = "Green North-South" if action == 0 else "Green East-West"
                print(f"Step {step} | Agent '{agent_id}' | State: [Local: {int(obs[0])}, Upstream: {int(obs[1])}] | Action Selected: {phase_name}")
                
            env.step(action)
            
            # Simple heuristic to delay visually roughly once per environment step (4 agents = 4 actions = 1 step)
            if action is not None and step % 4 == 0:
                time.sleep(args.delay)
            
            step += 1
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        env.close()
        print("-" * 100)
        print("SUMO-GUI session finished successfully.")

if __name__ == "__main__":
    run_gui_simulation()
