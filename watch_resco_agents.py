import os
import sys
import time
import argparse
import numpy as np

from simulator.env_setup_resco import make_resco_env
from marl_algorithms import TabularQLearningAgent, HystereticQLearningAgent
from watch_agents import FixedTimeController

def parse_args():
    parser = argparse.ArgumentParser(description="Watch pre-trained MARL agents in SUMO-GUI on RESCO scenarios.")
    parser.add_argument("--scenario", type=str, default="cologne3",
                        choices=["cologne3", "grid4x4"],
                        help="RESCO scenario to visualize.")
    parser.add_argument("--algo", type=str, default="iql_tabular",
                        choices=["iql_tabular", "hysteretic", "fixed"],
                        help="Algorithm to watch in GUI.")
    parser.add_argument("--delay", type=float, default=0.1, help="Simulation step sleep delay in seconds.")
    return parser.parse_args()

def load_policy(algo, scenario, agent_ids):
    """Loads a pre-trained policy for each agent from disk."""
    agents = {}
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    for agent_id in agent_ids:
        # Sanitize agent ID for filename loading
        sanitized_id = str(agent_id).replace(":", "_").replace("/", "_")
        
        if algo == "iql_tabular":
            agent = TabularQLearningAgent(agent_id, num_states=625)
            path = os.path.join(model_dir, f"{algo}_{scenario}_{sanitized_id}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded tabular Q-table for agent {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with un-trained/empty policy.")
            agents[agent_id] = agent
            
        elif algo == "hysteretic":
            agent = HystereticQLearningAgent(agent_id, num_states=625)
            path = os.path.join(model_dir, f"{algo}_{scenario}_{sanitized_id}.npy")
            if os.path.exists(path):
                agent.q_table = np.load(path)
                print(f"Loaded Hysteretic Q-table for agent {agent_id} from {path}")
            else:
                print(f"Warning: {path} not found. Running with un-trained/empty policy.")
            agents[agent_id] = agent
            
        elif algo == "fixed":
            agent = FixedTimeController(agent_id, ew_steps=10, ns_steps=10, offset_steps=0)
            print(f"Loaded Fixed-Time Controller (50/50) for agent {agent_id}")
            agents[agent_id] = agent
                
    return agents

def run_gui_simulation():
    args = parse_args()
    
    print(f"\nLaunching SUMO-GUI for algorithm '{args.algo}' on the '{args.scenario}' Scenario...")
    
    # Initialize the environment with GUI
    env = make_resco_env(args.scenario, num_seconds=3600, use_gui=True)
    env.reset()
    
    agent_ids = env.possible_agents
    agents = load_policy(args.algo, args.scenario, agent_ids)
    
    print("\nStarting Real-time Decision Monitoring...")
    print("-" * 100)
    
    step = 0
    num_agents = len(agent_ids)
    
    try:
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                agent = agents.get(agent_id)
                obs = np.array(observation, dtype=np.float32)
                action = agent.compute_action(obs, explore=False)
                
                phase_name = "Green North-South" if action == 0 else "Green East-West"
                print(f"Step {step} | Agent '{agent_id}' | State: [Local EW: {int(obs[0])}, NS: {int(obs[1])}] | Action: {phase_name}")
                
            env.step(action)
            
            # Simple heuristic to delay visually roughly once per environment step
            if action is not None and step % num_agents == 0:
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
