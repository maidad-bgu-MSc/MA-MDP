import os
import numpy as np
from tqdm import tqdm
from simulator.env_setup import make_wave_env
from marl_algorithms import TabularQLearningAgent, HystereticQLearningAgent
from watch_agents import FixedTimeController

def train_tabular_agents(algo="iql_tabular", episodes=100, sim_seconds=1000, eval_interval=10, eval_seconds=600):
    print("\n" + "="*50)
    print(f"TRAINING {algo.upper()} AGENTS ({episodes} episodes)")
    print("="*50)
    
    # Initialize Environment
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    if not os.path.exists(net_file) or not os.path.exists(rou_file):
        from simulator.generate_1x4_wave import build_1x4_scenario
        build_1x4_scenario()
        
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds)
    env.reset()
    
    if algo == "iql_tabular":
        agents = {agent: TabularQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    elif algo == "hysteretic":
        agents = {agent: HystereticQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    else:
        raise ValueError(f"Unsupported tabular algorithm: {algo}")
    
    eval_history = []
    
    # Progress Bar for Episodes
    for episode in tqdm(range(episodes), desc=f"{algo} Training Progress"):
        env.reset()
        last_obs = {a: None for a in env.possible_agents}
        last_action = {a: None for a in env.possible_agents}
        
        # Epsilon decay
        epsilon = max(0.05, 1.0 - (episode / (episodes * 0.8)))
        for agent_id in agents:
            agents[agent_id].epsilon = epsilon
            
        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs = np.array(observation, dtype=np.float32)
            
            # Q-learning update step
            if last_obs[agent_id] is not None:
                agents[agent_id].update(
                    obs=last_obs[agent_id], 
                    action=last_action[agent_id], 
                    reward=reward, 
                    next_obs=obs, 
                    done=termination or truncation,
                )
                
            if termination or truncation:
                action = None
            else:
                action = agents[agent_id].compute_action(obs, explore=True)
                last_obs[agent_id] = obs
                last_action[agent_id] = action
                
            env.step(action)
            
        # Periodic Evaluation
        if (episode + 1) % eval_interval == 0:
            # We use evaluate_agents inside the same file.
            # evaluate_agents creates its own fresh environment to avoid messing with the training env.
            reward = evaluate_agents(agents, algo_name=f"{algo} (Epoch {episode+1})", sim_seconds=eval_seconds)
            eval_history.append((episode + 1, reward))
            
    # Save the trained models
    os.makedirs("models", exist_ok=True)
    for agent_id, agent in agents.items():
        path = os.path.join("models", f"{algo}_{agent_id}.npy")
        np.save(path, agent.q_table)
        
    print(f"\nTraining completed. Q-tables saved to models/{algo}_*.npy")
    env.close()
    return agents, eval_history

def evaluate_agents(agents_dict, algo_name="Tabular IQL", sim_seconds=3600):
    env = make_wave_env(net_file="wave_1x4.net.xml", route_file="wave_1x4.rou.xml", num_seconds=sim_seconds)
    env.reset()
    
    total_rewards = {a: 0.0 for a in env.possible_agents}
    
    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        total_rewards[agent_id] += reward
        
        if termination or truncation:
            action = None
        else:
            obs = np.array(observation, dtype=np.float32)
            action = agents_dict[agent_id].compute_action(obs, explore=False)
            
        env.step(action)
        
    env.close()
    
    # In PettingZoo, the rewards returned are the marginal rewards since last action.
    # The GlobalRewardWrapper broadcasts the same global reward to all agents.
    # Therefore, we can just look at the sum for any one agent (or average them) to get the true episode return.
    system_reward = sum(total_rewards.values()) / len(total_rewards)
    
    print(f"--- {algo_name} Evaluation ---")
    print(f"System Global Return (Total Negative Delay): {system_reward:.2f}")
    return system_reward

import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Train Tabular Agents (100 episodes, evaluate every 10)
    trained_iql_agents, iql_history = train_tabular_agents(algo="iql_tabular", episodes=100, sim_seconds=1000, eval_interval=5, eval_seconds=600)
    trained_hysteretic_agents, hyst_history = train_tabular_agents(algo="hysteretic", episodes=100, sim_seconds=1000, eval_interval=5, eval_seconds=600)
    
    print("\n" + "="*50)
    print("EVALUATING FIXED BASELINE (600 seconds)")
    print("="*50)
    
    # 2. Evaluate Fixed Time Baseline (using 600s to match the training eval duration)
    # Empirically determined to be the best fixed baseline: 50/50 split, no offsets
    fixed_agents = {
        agent: FixedTimeController(agent, ew_steps=10, ns_steps=10, offset_steps=0)
        for agent in trained_iql_agents.keys()
    }
    fixed_reward = evaluate_agents(fixed_agents, algo_name="Fixed-Time (50/50)", sim_seconds=600)
    
    # 3. Save to CSV
    log_file = "training_evaluation_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Tabular_IQL_Reward", "Hysteretic_Reward", "Fixed_Baseline_Reward"])
        for (ep, iql_r), (_, hyst_r) in zip(iql_history, hyst_history):
            writer.writerow([ep, iql_r, hyst_r, fixed_reward])
            
    print(f"\nSaved evaluation history to {log_file}")
    
    # 4. Plot Learning Curves (Log-Scaled Absolute Delay)
    epochs = [x[0] for x in iql_history]
    iql_delays = [abs(x[1]) for x in iql_history]
    hyst_delays = [abs(x[1]) for x in hyst_history]
    fixed_delay = abs(fixed_reward)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, iql_delays, marker='o', label="Tabular IQL")
    plt.plot(epochs, hyst_delays, marker='s', label="Hysteretic Q-Learning")
    plt.axhline(y=fixed_delay, color='r', linestyle='--', label="Fixed-Time (50/50) Baseline")
    
    plt.yscale('log')
    plt.xlabel("Training Epochs")
    plt.ylabel("Absolute Total Delay (Log Scale) [Lower is Better]")
    plt.title("MARL Coordination: Learning Curves (600s Evaluation)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig("learning_curves.png", dpi=300)
    print("Saved learning curves plot to learning_curves.png")
