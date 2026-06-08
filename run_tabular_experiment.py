import os
import csv
import sys
import traceback
from datetime import datetime
import numpy as np
from tqdm import tqdm
from simulator.env_setup import make_wave_env, GlobalRewardWrapper, QueueObservationFunction, global_reward_fn
from marl_algorithms import TabularQLearningAgent, HystereticQLearningAgent, TabularVDNAgents
from watch_agents import FixedTimeController
from simulator.problem_generator import SCENARIOS, generate_problem, make_problem_env, make_problem_parallel_env
from seeding import episode_seed, eval_seed


class _Tee:
    """Duplicates writes to multiple streams. Used to mirror stdout/stderr into a log file
    so a mid-run crash leaves a full trace on disk (SUMO TraCI disconnects, etc.)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _open_log(path="training.log"):
    fh = open(path, "w", buffering=1, encoding="utf-8")
    fh.write(f"=== run_tabular_experiment.py started at {datetime.now().isoformat()} ===\n")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh


def train_tabular_agents(algo="iql_tabular", scenario_name="baseline", episodes=100,
                         sim_seconds=1000, eval_interval=10, eval_seconds=600, seed=None):
    print("\n" + "="*50)
    print(f"TRAINING {algo.upper()} [{scenario_name}] ({episodes} episodes, seed={seed})")
    print("="*50)

    net_file, rou_file = generate_problem(scenario_name)
    construct_seed = seed if seed is not None else "random"
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds,
                        sumo_seed=construct_seed)
    env.reset()

    if algo == "iql_tabular":
        agents = {agent: TabularQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    elif algo == "hysteretic":
        agents = {agent: HystereticQLearningAgent(agent, num_states=625) for agent in env.possible_agents}
    else:
        raise ValueError(f"Unsupported tabular algorithm: {algo}")

    eval_history = []

    for episode in tqdm(range(episodes), desc=f"{algo} [{scenario_name}]"):
        env.reset(seed=episode_seed(seed, episode) if seed is not None else None)
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
            es = eval_seed(seed) if seed is not None else None
            r = evaluate_agents(agents, algo_name=f"{algo} [{scenario_name}] (ep {episode+1})",
                                net_file=net_file, rou_file=rou_file, sim_seconds=eval_seconds,
                                eval_sumo_seed=es)
            eval_history.append((episode + 1, r))

    model_dir = os.path.join("models", f"seed{seed}") if seed is not None else "models"
    os.makedirs(model_dir, exist_ok=True)
    for agent_id, agent in agents.items():
        np.save(os.path.join(model_dir, f"{algo}_{scenario_name}_{agent_id}.npy"), agent.q_table)

    print(f"\nTraining complete. Models saved to {model_dir}/{algo}_{scenario_name}_*.npy")
    env.close()
    return agents, eval_history


def evaluate_agents(agents_dict, algo_name="Tabular IQL", sim_seconds=3600,
                    net_file="wave_1x4.net.xml", rou_file="wave_1x4.rou.xml", eval_sumo_seed=None):
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=sim_seconds,
                        sumo_seed=construct_seed)
    env.reset(seed=eval_sumo_seed)

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
                     eval_interval=10, eval_seconds=600, seed=None):
    print("\n" + "="*50)
    print(f"TRAINING VDN [{scenario_name}] ({episodes} episodes, seed={seed})")
    print("="*50)

    net_file, rou_file = generate_problem(scenario_name)
    construct_seed = seed if seed is not None else "random"
    env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
    agent_ids = env.possible_agents
    vdn = TabularVDNAgents(agent_ids=agent_ids, num_states=625)
    eval_history = []

    for episode in tqdm(range(episodes), desc=f"VDN [{scenario_name}]"):
        vdn.epsilon = max(0.05, 1.0 - (episode / (episodes * 0.8)))
        obs_dict, _ = env.reset(seed=episode_seed(seed, episode) if seed is not None else None)

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
            es = eval_seed(seed) if seed is not None else None
            r = evaluate_vdn_agents(vdn, scenario_name=scenario_name,
                                    net_file=net_file, rou_file=rou_file, sim_seconds=eval_seconds,
                                    eval_sumo_seed=es)
            eval_history.append((episode + 1, r))

    env.close()

    model_dir = os.path.join("models", f"seed{seed}") if seed is not None else "models"
    os.makedirs(model_dir, exist_ok=True)
    for aid, qt in vdn.q_tables.items():
        np.save(os.path.join(model_dir, f"vdn_{scenario_name}_{aid}.npy"), qt)
    print(f"\nVDN training complete. Models saved to {model_dir}/vdn_{scenario_name}_*.npy")
    return vdn, eval_history


def evaluate_vdn_agents(vdn, scenario_name="baseline", sim_seconds=3600,
                         net_file=None, rou_file=None, eval_sumo_seed=None):
    if net_file is None or rou_file is None:
        net_file, rou_file = generate_problem(scenario_name)
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
    agent_ids = env.possible_agents
    obs_dict, _ = env.reset(seed=eval_sumo_seed)
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


def run_scenario(scenario, episodes=100, eval_interval=5, seed=None, out_dir="outputs"):
    """Train IQL/Hysteretic/VDN on a 1x4 scenario, evaluate against the coordinated
    fixed-time baseline, write a per-(scenario, seed) CSV, and return the histories.
    Reused by run_seeded_experiment.py."""
    print(f"\n{'='*60}\nSCENARIO: {scenario.upper()}  (seed={seed})\n{'='*60}")
    net_file, rou_file = generate_problem(scenario)

    _, iql_h = train_tabular_agents(algo="iql_tabular", scenario_name=scenario,
                                    episodes=episodes, eval_interval=eval_interval, seed=seed)
    _, hyst_h = train_tabular_agents(algo="hysteretic", scenario_name=scenario,
                                     episodes=episodes, eval_interval=eval_interval, seed=seed)
    _, vdn_h = train_vdn_agents(scenario_name=scenario, episodes=episodes,
                                eval_interval=eval_interval, seed=seed)

    # 1x4 signals are 2-phase (EW/NS), so the coordinated fixed-time controller is the
    # natural baseline. Evaluate it on the same per-run eval seed for a fair comparison.
    es = eval_seed(seed) if seed is not None else None
    fixed_agents = {aid: FixedTimeController(aid, ew_steps=10, ns_steps=10, offset_steps=0)
                    for aid in ["A0", "B0", "C0", "D0"]}
    fixed_reward = evaluate_agents(fixed_agents, algo_name=f"Fixed-Time [{scenario}]",
                                   net_file=net_file, rou_file=rou_file, sim_seconds=600,
                                   eval_sumo_seed=es)

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_seed{seed}" if seed is not None else ""
    log_file = os.path.join(out_dir, f"training_evaluation_log_{scenario}{suffix}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Tabular_IQL_Reward", "Hysteretic_Reward", "VDN_Reward", "Fixed_Baseline_Reward"])
        min_len = min(len(iql_h), len(hyst_h), len(vdn_h))
        for i in range(min_len):
            ep = iql_h[i][0]
            writer.writerow([ep, iql_h[i][1], hyst_h[i][1], vdn_h[i][1], fixed_reward])
    print(f"Saved {log_file}")
    return {"iql": iql_h, "hyst": hyst_h, "vdn": vdn_h, "fixed": fixed_reward, "csv": log_file}


if __name__ == "__main__":
    EPISODES = 100
    EVAL_INTERVAL = 5

    log_fh = _open_log("training.log")
    current_phase = "startup"
    try:
        for scenario in SCENARIOS:
            current_phase = scenario
            run_scenario(scenario, episodes=EPISODES, eval_interval=EVAL_INTERVAL, seed=None)
    except BaseException:
        print(f"\n!!! CRASHED during phase: {current_phase} at {datetime.now().isoformat()} !!!")
        traceback.print_exc()
        raise
    finally:
        print(f"=== run_tabular_experiment.py finished at {datetime.now().isoformat()} ===")
        log_fh.close()
