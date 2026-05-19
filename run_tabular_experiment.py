import os
import numpy as np
from tqdm import tqdm
from simulator.env_setup import make_wave_env
from marl_algorithms import TabularQLearningAgent
from watch_agents import FixedTimeController

def train_tabular_agents(episodes=30, sim_seconds=1000):
    print("\n" + "="*50)
    print(f"TRAINING TABULAR Q-LEARNING AGENTS ({episodes} episodes)")
    print("="*50)
    
    # Initialize Environment
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    if not os.path.exists(net_file) or not os.path.exists(rou_file):
        from simulator.generate_1x4_wave import build_1x4_scenario
        build_1x4_scenario()
        
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds)
    env.reset()
    
    agents = {agent: TabularQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    
    # Progress Bar for Episodes
    for episode in tqdm(range(episodes), desc="Training Progress"):
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
                    done=termination or truncation
                )
                
            if termination or truncation:
                action = None
            else:
                action = agents[agent_id].compute_action(obs, explore=True)
                last_obs[agent_id] = obs
                last_action[agent_id] = action
                
            env.step(action)
            
    # Save the trained models
    os.makedirs("models", exist_ok=True)
    for agent_id, agent in agents.items():
        path = os.path.join("models", f"iql_tabular_{agent_id}.npy")
        np.save(path, agent.q_table)
        
    print("\nTraining completed. Q-tables saved to models/")
    env.close()
    return agents

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

if __name__ == "__main__":
    # 1. Train Tabular Agents
    trained_iql_agents = train_tabular_agents(episodes=30, sim_seconds=1000)
    
    print("\n" + "="*50)
    print("="*50)
    
    # 2. Evaluate Trained IQL
    iql_reward = evaluate_agents(trained_iql_agents, algo_name="Tabular Q-Learning", sim_seconds=3600)
    
    # 3. Evaluate Fixed Time Baseline
    print("STARTING DETERMINISTIC EVALUATION (3600 seconds)")
    fixed_agents = {agent: FixedTimeController(agent, cycle_steps=6) for agent in trained_iql_agents.keys()}
    fixed_reward = evaluate_agents(fixed_agents, algo_name="Fixed-Time Controller", sim_seconds=3600)
    
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"Fixed-Time Total Reward: {fixed_reward:.2f}")
    print(f"Tabular IQL Total Reward: {iql_reward:.2f}")
    
    if iql_reward > fixed_reward:
        improvement = ((iql_reward - fixed_reward) / abs(fixed_reward)) * 100
        print(f"\nResult: Tabular IQL outperformed Fixed-Time by {improvement:.1f}%!")
    else:
        print("\nResult: Tabular IQL did not outperform Fixed-Time. (Try more training episodes!)")
