import os
import csv
import sys
import traceback
from datetime import datetime
import numpy as np
from tqdm import tqdm

from simulator.env_setup_resco import make_resco_env, make_resco_parallel_env
from marl_algorithms import (TabularQLearningAgent, HystereticQLearningAgent, TabularVDNAgents,
                             get_resco_state, RESCO_NUM_STATES)
from seeding import episode_seed, eval_seed


class RoundRobinFixedController:
    """Fixed-time baseline that cycles through ALL green phases of a junction in order,
    holding each for a fixed number of decision steps.

    This is the natural fixed-time plan for real multi-phase RESCO junctions (cologne3
    has 3-4 phases, grid4x4 has 8). The 1x4 EW/NS FixedTimeController only ever exercises
    2 phases, which starves the other movements on multi-phase junctions.
    """
    def __init__(self, agent_id, num_actions, steps_per_phase=2):
        self.agent_id = agent_id
        self.num_actions = int(num_actions)
        self.steps_per_phase = max(1, int(steps_per_phase))
        self.step_count = 0

    def compute_action(self, obs, explore=False):
        phase = (self.step_count // self.steps_per_phase) % self.num_actions
        self.step_count += 1
        return int(phase)


class _Tee:
    """Duplicates writes to multiple streams. Used to mirror stdout/stderr into a log file
    so a mid-run crash leaves a full trace on disk."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _open_log(path="resco_tabular_training.log"):
    fh = open(path, "w", buffering=1, encoding="utf-8")
    fh.write(f"=== run_resco_tabular_experiment.py started at {datetime.now().isoformat()} ===\n")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh


def train_tabular_agents(algo="iql_tabular", scenario_name="cologne3", episodes=100,
                         sim_seconds=1000, eval_interval=10, eval_seconds=600, seed=None):
    print("\n" + "="*50)
    print(f"TRAINING {algo.upper()} [{scenario_name}] ({episodes} episodes, seed={seed})")
    print("="*50)

    construct_seed = seed if seed is not None else "random"
    env = make_resco_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
    env.reset()

    # Real RESCO junctions have heterogeneous green-phase counts; size each agent's
    # action space to its own junction (cologne3: 3-4, grid4x4: 8) instead of assuming 2.
    action_counts = {agent: env.action_space(agent).n for agent in env.possible_agents}

    # Phase-aware state encoding (get_resco_state / RESCO_NUM_STATES) so the table can map
    # the most-demanded phase onto the phase-index action on multi-phase RESCO junctions.
    if algo == "iql_tabular":
        agents = {agent: TabularQLearningAgent(agent, num_states=RESCO_NUM_STATES,
                                               num_actions=action_counts[agent],
                                               state_encoder=get_resco_state)
                  for agent in env.possible_agents}
    elif algo == "hysteretic":
        agents = {agent: HystereticQLearningAgent(agent, num_states=RESCO_NUM_STATES,
                                                  num_actions=action_counts[agent],
                                                  state_encoder=get_resco_state)
                  for agent in env.possible_agents}
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
                                scenario_name=scenario_name, sim_seconds=eval_seconds,
                                eval_sumo_seed=es)
            eval_history.append((episode + 1, r))

    model_dir = os.path.join("models", f"seed{seed}") if seed is not None else "models"
    os.makedirs(model_dir, exist_ok=True)
    for agent_id, agent in agents.items():
        # Sanitize agent ID for filename safety (real-world OSM junction IDs can contain special characters)
        sanitized_id = str(agent_id).replace(":", "_").replace("/", "_")
        np.save(os.path.join(model_dir, f"{algo}_{scenario_name}_{sanitized_id}.npy"), agent.q_table)

    print(f"\nTraining complete. Models saved to {model_dir}/{algo}_{scenario_name}_*.npy")
    env.close()
    return agents, eval_history


def evaluate_agents(agents_dict, algo_name="Tabular IQL", sim_seconds=3600,
                    scenario_name="cologne3", eval_sumo_seed=None):
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_resco_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
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


def train_vdn_agents(scenario_name="cologne3", episodes=100, sim_seconds=1000,
                     eval_interval=10, eval_seconds=600, seed=None):
    print("\n" + "="*50)
    print(f"TRAINING VDN [{scenario_name}] ({episodes} episodes, seed={seed})")
    print("="*50)

    construct_seed = seed if seed is not None else "random"
    env = make_resco_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
    agent_ids = env.possible_agents
    # Per-agent action counts for heterogeneous RESCO junctions.
    action_counts = {aid: env.action_space(aid).n for aid in agent_ids}
    vdn = TabularVDNAgents(agent_ids=agent_ids, num_states=RESCO_NUM_STATES,
                           num_actions=action_counts, state_encoder=get_resco_state)
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
            r = evaluate_vdn_agents(vdn, scenario_name=scenario_name, sim_seconds=eval_seconds,
                                    eval_sumo_seed=es)
            eval_history.append((episode + 1, r))

    env.close()

    model_dir = os.path.join("models", f"seed{seed}") if seed is not None else "models"
    os.makedirs(model_dir, exist_ok=True)
    for aid, qt in vdn.q_tables.items():
        sanitized_id = str(aid).replace(":", "_").replace("/", "_")
        np.save(os.path.join(model_dir, f"vdn_{scenario_name}_{sanitized_id}.npy"), qt)
    print(f"\nVDN training complete. Models saved to {model_dir}/vdn_{scenario_name}_*.npy")
    return vdn, eval_history


def evaluate_vdn_agents(vdn, scenario_name="cologne3", sim_seconds=3600, eval_sumo_seed=None):
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_resco_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
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


def evaluate_fixed_native(scenario_name="cologne3", sim_seconds=600, eval_sumo_seed=None):
    """SUMO-native fixed-time baseline: every junction follows the signal program defined
    in its .net.xml (fixed_ts=True) and ignores agent actions. This is the canonical
    fixed-time plan the RESCO benchmark itself compares against."""
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_resco_parallel_env(scenario_name, num_seconds=sim_seconds,
                                  sumo_seed=construct_seed, fixed_ts=True)
    agent_ids = env.possible_agents
    env.reset(seed=eval_sumo_seed)
    total_reward = 0.0

    while True:
        # Actions are ignored under fixed_ts; pass zeros to satisfy the step API.
        actions = {aid: 0 for aid in agent_ids}
        _, rewards, terminations, truncations, _ = env.step(actions)
        vals = list(rewards.values())
        if vals:
            total_reward += float(vals[0])
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()
    print(f"--- Fixed-Native [{scenario_name}] | Return: {total_reward:.2f}")
    return total_reward


def evaluate_fixed_roundrobin(scenario_name="cologne3", sim_seconds=600, eval_sumo_seed=None):
    """Round-robin fixed-time baseline: cycles through ALL green phases of every junction."""
    temp_env = make_resco_env(scenario_name, num_seconds=200)
    action_counts = {aid: temp_env.action_space(aid).n for aid in temp_env.possible_agents}
    temp_env.close()
    rr_agents = {aid: RoundRobinFixedController(aid, num_actions=action_counts[aid])
                 for aid in action_counts}
    return evaluate_agents(rr_agents, algo_name=f"Fixed-RoundRobin [{scenario_name}]",
                           scenario_name=scenario_name, sim_seconds=sim_seconds,
                           eval_sumo_seed=eval_sumo_seed)


def run_resco_scenario(scenario, episodes=100, eval_interval=5, seed=None, out_dir="outputs"):
    """Train IQL/Hysteretic/VDN on a RESCO scenario and evaluate against both fixed-time
    baselines (SUMO-native + round-robin). Writes a per-(scenario, seed) CSV and returns
    the histories. Reused by run_seeded_experiment.py."""
    print(f"\n{'='*60}\nSCENARIO: {scenario.upper()}  (seed={seed})\n{'='*60}")

    _, iql_h = train_tabular_agents(algo="iql_tabular", scenario_name=scenario,
                                    episodes=episodes, eval_interval=eval_interval, seed=seed)
    _, hyst_h = train_tabular_agents(algo="hysteretic", scenario_name=scenario,
                                     episodes=episodes, eval_interval=eval_interval, seed=seed)
    _, vdn_h = train_vdn_agents(scenario_name=scenario, episodes=episodes,
                                eval_interval=eval_interval, seed=seed)

    es = eval_seed(seed) if seed is not None else None
    fixed_native = evaluate_fixed_native(scenario_name=scenario, sim_seconds=600, eval_sumo_seed=es)
    fixed_rr = evaluate_fixed_roundrobin(scenario_name=scenario, sim_seconds=600, eval_sumo_seed=es)

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_seed{seed}" if seed is not None else ""
    log_file = os.path.join(out_dir, f"training_evaluation_log_{scenario}{suffix}.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Tabular_IQL_Reward", "Hysteretic_Reward", "VDN_Reward",
                         "Fixed_Native_Reward", "Fixed_RoundRobin_Reward"])
        min_len = min(len(iql_h), len(hyst_h), len(vdn_h))
        for i in range(min_len):
            ep = iql_h[i][0]
            writer.writerow([ep, iql_h[i][1], hyst_h[i][1], vdn_h[i][1], fixed_native, fixed_rr])
    print(f"Saved {log_file}")
    return {"iql": iql_h, "hyst": hyst_h, "vdn": vdn_h,
            "fixed_native": fixed_native, "fixed_roundrobin": fixed_rr, "csv": log_file}


if __name__ == "__main__":
    EPISODES = 100
    EVAL_INTERVAL = 5
    RESCO_SCENARIOS = ["cologne3", "grid4x4"]

    log_fh = _open_log("resco_tabular_training.log")
    current_phase = "startup"
    try:
        for scenario in RESCO_SCENARIOS:
            current_phase = scenario
            run_resco_scenario(scenario, episodes=EPISODES, eval_interval=EVAL_INTERVAL, seed=None)
    except BaseException:
        print(f"\n!!! CRASHED during phase: {current_phase} at {datetime.now().isoformat()} !!!")
        traceback.print_exc()
        raise
    finally:
        try:
            print("\nGenerating evaluation plots for RESCO scenarios...")
            from plot_resco_results import plot_resco_tabular_learning_curves, plot_resco_cross_algorithm_bar
            plot_resco_tabular_learning_curves()
            plot_resco_cross_algorithm_bar()
        except Exception as e:
            print(f"Plotting failed: {e}")
        print(f"=== run_resco_tabular_experiment.py finished at {datetime.now().isoformat()} ===")
        log_fh.close()
