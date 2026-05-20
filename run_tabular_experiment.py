import os
import csv
import numpy as np
from tqdm import tqdm
from simulator.env_setup import make_wave_env, GlobalRewardWrapper, QueueObservationFunction, global_reward_fn
from marl_algorithms import TabularQLearningAgent, HystereticQLearningAgent, TabularVDNAgents
from watch_agents import FixedTimeController
from simulator.problem_generator import SCENARIOS, generate_problem, make_problem_env, make_problem_parallel_env


def train_tabular_agents(algo="iql_tabular", scenario_name="baseline", episodes=100,
                         sim_seconds=1000, eval_interval=10, eval_seconds=600):
    print("\n" + "="*50)
    print(f"TRAINING {algo.upper()} [{scenario_name}] ({episodes} episodes)")
    print("="*50)

    net_file, rou_file = generate_problem(scenario_name)
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds)
    env.reset()

    if algo == "iql_tabular":
        agents = {agent: TabularQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    elif algo == "hysteretic":
        agents = {agent: HystereticQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    else:
        raise ValueError(f"Unsupported tabular algorithm: {algo}")

    eval_history = []

    for episode in tqdm(range(episodes), desc=f"{algo} [{scenario_name}]"):
        env.reset()
        last_obs = {a: None for a in env.possible_agents}
        last_action = {a: None for a in env.possible_agents}

        epsilon = max(0.05, 1.0 - (episode / (episodes * 0.8)))
        for agent_id in agents:
            agents[agent_id].epsilon = epsilon

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            obs = np.array(observation, dtype=np.float32)

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

        if (episode + 1) % eval_interval == 0:
            r = evaluate_agents(agents, algo_name=f"{algo} [{scenario_name}] (ep {episode+1})",
                                net_file=net_file, rou_file=rou_file, sim_seconds=eval_seconds)
            eval_history.append((episode + 1, r))

    os.makedirs("models", exist_ok=True)
    for agent_id, agent in agents.items():
        np.save(os.path.join("models", f"{algo}_{scenario_name}_{agent_id}.npy"), agent.q_table)

    print(f"\nTraining complete. Models saved to models/{algo}_{scenario_name}_*.npy")
    env.close()
    return agents, eval_history


def evaluate_agents(agents_dict, algo_name="Tabular IQL", sim_seconds=3600,
                    net_file="wave_1x4.net.xml", rou_file="wave_1x4.rou.xml"):
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds)
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
    system_reward = sum(total_rewards.values()) / len(total_rewards)
    print(f"--- {algo_name} | Return: {system_reward:.2f}")
    return system_reward


def train_vdn_agents(scenario_name="baseline", episodes=100, sim_seconds=1000,
                     eval_interval=10, eval_seconds=600):
    print("\n" + "="*50)
    print(f"TRAINING VDN [{scenario_name}] ({episodes} episodes)")
    print("="*50)

    net_file, rou_file = generate_problem(scenario_name)
    env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds)
    agent_ids = env.possible_agents
    vdn = TabularVDNAgents(agent_ids=agent_ids, num_states=625)
    eval_history = []

    for episode in tqdm(range(episodes), desc=f"VDN [{scenario_name}]"):
        vdn.epsilon = max(0.05, 1.0 - (episode / (episodes * 0.8)))
        obs_dict, _ = env.reset()

        while True:
            actions = {
                aid: vdn.compute_action(aid, np.array(obs_dict[aid], dtype=np.float32), explore=True)
                for aid in agent_ids
            }
            next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
            reward = float(list(rewards.values())[0])
            done = any(terminations.values()) or any(truncations.values())

            vdn.update(
                obs_dict={aid: np.array(obs_dict[aid], dtype=np.float32) for aid in agent_ids},
                action_dict=actions,
                reward=reward,
                next_obs_dict={aid: np.array(next_obs_dict[aid], dtype=np.float32) for aid in agent_ids},
                done=done,
            )
            obs_dict = next_obs_dict
            if done:
                break

        if (episode + 1) % eval_interval == 0:
            r = evaluate_vdn_agents(vdn, scenario_name=scenario_name,
                                    net_file=net_file, rou_file=rou_file, sim_seconds=eval_seconds)
            eval_history.append((episode + 1, r))

    env.close()

    os.makedirs("models", exist_ok=True)
    for aid, qt in vdn.q_tables.items():
        np.save(os.path.join("models", f"vdn_{scenario_name}_{aid}.npy"), qt)
    print(f"\nVDN training complete. Models saved to models/vdn_{scenario_name}_*.npy")
    return vdn, eval_history


def evaluate_vdn_agents(vdn, scenario_name="baseline", sim_seconds=3600,
                         net_file=None, rou_file=None):
    if net_file is None or rou_file is None:
        net_file, rou_file = generate_problem(scenario_name)
    env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds)
    agent_ids = env.possible_agents
    obs_dict, _ = env.reset()
    total_reward = 0.0

    while True:
        actions = {
            aid: vdn.compute_action(aid, np.array(obs_dict[aid], dtype=np.float32), explore=False)
            for aid in agent_ids
        }
        next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += float(list(rewards.values())[0])
        obs_dict = next_obs_dict
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()
    print(f"--- VDN [{scenario_name}] | Return: {total_reward:.2f}")
    return total_reward


if __name__ == "__main__":
    EPISODES = 100
    EVAL_INTERVAL = 5

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*60}")

        net_file, rou_file = generate_problem(scenario)

        _, iql_h = train_tabular_agents(
            algo="iql_tabular", scenario_name=scenario,
            episodes=EPISODES, eval_interval=EVAL_INTERVAL
        )
        _, hyst_h = train_tabular_agents(
            algo="hysteretic", scenario_name=scenario,
            episodes=EPISODES, eval_interval=EVAL_INTERVAL
        )
        _, vdn_h = train_vdn_agents(
            scenario_name=scenario, episodes=EPISODES, eval_interval=EVAL_INTERVAL
        )

        # Fixed baseline for this scenario
        fixed_agents = {
            aid: FixedTimeController(aid, ew_steps=10, ns_steps=10, offset_steps=0)
            for aid in ["A0", "B0", "C0", "D0"]
        }
        fixed_reward = evaluate_agents(
            fixed_agents, algo_name=f"Fixed-Time [{scenario}]",
            net_file=net_file, rou_file=rou_file, sim_seconds=600
        )

        log_file = f"training_evaluation_log_{scenario}.csv"
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Tabular_IQL_Reward", "Hysteretic_Reward", "VDN_Reward", "Fixed_Baseline_Reward"])
            min_len = min(len(iql_h), len(hyst_h), len(vdn_h))
            for i in range(min_len):
                ep = iql_h[i][0]
                writer.writerow([ep, iql_h[i][1], hyst_h[i][1], vdn_h[i][1], fixed_reward])
        print(f"Saved {log_file}")
