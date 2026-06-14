import os
import copy
import csv
import sys
import traceback
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from marl_algorithms import QMIXAgentNetwork, QMIXMixingNetwork
from simulator.problem_generator import SCENARIOS
from seeding import episode_seed, eval_seed

# RESCO scenarios use a different env maker (real multi-phase junctions, heterogeneous
# per-agent action counts) and a different observation function. Everything downstream
# (agent count, obs dim, per-agent action dims, mixer state dim) is derived from the env
# at run time, so the same QMIX code trains on both the 1x4 corridor and RESCO networks.
RESCO_SCENARIOS = ("cologne3", "grid4x4")


def make_parallel_env(scenario_name, num_seconds, sumo_seed):
    """Return the appropriate PettingZoo parallel env for a scenario (1x4 vs RESCO)."""
    if scenario_name in RESCO_SCENARIOS:
        from simulator.env_setup_resco import make_resco_parallel_env
        return make_resco_parallel_env(scenario_name, num_seconds=num_seconds, sumo_seed=sumo_seed)
    from simulator.problem_generator import make_problem_parallel_env
    return make_problem_parallel_env(scenario_name, num_seconds=num_seconds, sumo_seed=sumo_seed)


class _Tee:
    """Duplicates writes to multiple streams. Used to mirror stdout/stderr into a log file
    so a mid-run crash leaves a full trace on disk (SUMO TraCI disconnects, OOM, etc.)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _open_log(path="ctde_training.log"):
    fh = open(path, "w", buffering=1, encoding="utf-8")
    fh.write(f"=== run_ctde_experiment.py started at {datetime.now().isoformat()} ===\n")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh

# =====================================================================
# Replay Buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, state_dim, num_agents, capacity=10000):
        self.capacity = capacity
        self._obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, state_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, obs_flat, actions, reward, next_obs_flat, done):
        i = self._ptr % self.capacity
        self._obs[i] = obs_flat
        self._actions[i] = actions
        self._rewards[i] = reward
        self._next_obs[i] = next_obs_flat
        self._dones[i] = float(done)
        self._ptr += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.FloatTensor(self._obs[idx]),
            torch.LongTensor(self._actions[idx]),
            torch.FloatTensor(self._rewards[idx]),
            torch.FloatTensor(self._next_obs[idx]),
            torch.FloatTensor(self._dones[idx]),
        )

    @property
    def size(self):
        return self._size


# =====================================================================
# QMIX loss
# =====================================================================
def compute_qmix_loss(batch, agent_nets, target_agent_nets, mixer, target_mixer, gamma):
    obs_flat, actions, rewards, next_obs_flat, dones = batch
    B = obs_flat.shape[0]
    num_agents = len(agent_nets)
    obs_dim = obs_flat.shape[1] // num_agents

    # Split flat obs into per-agent observations. Agents may have different action-space
    # sizes (RESCO junctions have 3-8 phases); that is fine because each agent's Q-values
    # are gathered/maxed over its OWN network output below — only obs_dim must be uniform.
    obs_per_agent = obs_flat.view(B, num_agents, obs_dim)
    next_obs_per_agent = next_obs_flat.view(B, num_agents, obs_dim)

    # Chosen Q-values for each agent
    chosen_qs = []
    for i, net in enumerate(agent_nets):
        q_vals = net(obs_per_agent[:, i, :])          # (B, num_actions)
        chosen_q = q_vals.gather(1, actions[:, i].unsqueeze(1))  # (B, 1)
        chosen_qs.append(chosen_q)
    chosen_qs = torch.cat(chosen_qs, dim=1)           # (B, num_agents)

    # Mix current Q-values
    q_tot = mixer(chosen_qs, obs_flat)                # (B, 1)

    # Target: greedy max over next observations using target nets
    with torch.no_grad():
        target_qs = []
        for i, tnet in enumerate(target_agent_nets):
            next_q = tnet(next_obs_per_agent[:, i, :]).max(dim=1, keepdim=True)[0]  # (B, 1)
            target_qs.append(next_q)
        target_qs = torch.cat(target_qs, dim=1)       # (B, num_agents)
        q_tot_next = target_mixer(target_qs, next_obs_flat)  # (B, 1)
        y = rewards.unsqueeze(1) + gamma * q_tot_next * (1.0 - dones.unsqueeze(1))

    return nn.functional.mse_loss(q_tot, y)


# =====================================================================
# Training
# =====================================================================
def train_qmix(
    scenario_name,
    num_episodes=200,
    sim_seconds=1800,
    batch_size=32,
    buffer_capacity=10000,
    lr=1e-3,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_episodes=150,
    target_update_freq=10,
    min_buffer_size=500,
    eval_interval=20,
    eval_seconds=600,
    save_dir="models",
    seed=None,
):
    if seed is not None:
        save_dir = os.path.join(save_dir, f"seed{seed}")

    # Networks are built lazily on the first episode, once the env reveals the agent set,
    # observation dimension and each junction's action-space size. This makes the same code
    # work for the 1x4 corridor (4 agents x 2 actions) and RESCO nets (3-16 agents, 3-8
    # phases each) without any hard-coded shapes.
    agent_nets = target_agent_nets = mixer = target_mixer = None
    optimizer = buffer = None
    agent_ids = action_counts = None
    eval_history = []

    for episode in tqdm(range(num_episodes), desc=f"QMIX [{scenario_name}]"):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay_episodes)

        ep_seed = episode_seed(seed, episode) if seed is not None else "random"
        env = make_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=ep_seed)
        agent_ids = list(env.possible_agents)
        obs_dict, _ = env.reset(seed=(ep_seed if seed is not None else None))

        if agent_nets is None:
            num_agents = len(agent_ids)
            obs_dim = int(np.asarray(obs_dict[agent_ids[0]]).size)
            action_counts = [int(env.action_space(aid).n) for aid in agent_ids]
            state_dim = num_agents * obs_dim
            agent_nets = [QMIXAgentNetwork(obs_dim=obs_dim, action_dim=action_counts[i])
                          for i in range(num_agents)]
            target_agent_nets = [copy.deepcopy(net) for net in agent_nets]
            mixer = QMIXMixingNetwork(num_agents=num_agents, state_dim=state_dim)
            target_mixer = copy.deepcopy(mixer)
            all_params = list(mixer.parameters())
            for net in agent_nets:
                all_params += list(net.parameters())
            optimizer = optim.Adam(all_params, lr=lr)
            buffer = ReplayBuffer(state_dim=state_dim, num_agents=num_agents,
                                  capacity=buffer_capacity)
            print(f"  QMIX [{scenario_name}] {num_agents} agents, obs_dim={obs_dim}, "
                  f"action_counts={action_counts}")

        while True:
            obs_flat = np.concatenate([np.asarray(obs_dict[aid], dtype=np.float32) for aid in agent_ids])

            # Epsilon-greedy action selection (each agent samples within its own phase count)
            actions = {}
            with torch.no_grad():
                for i, aid in enumerate(agent_ids):
                    if np.random.rand() < epsilon:
                        actions[aid] = np.random.randint(action_counts[i])
                    else:
                        obs_t = torch.FloatTensor(np.asarray(obs_dict[aid], dtype=np.float32)).unsqueeze(0)
                        q_vals = agent_nets[i](obs_t)
                        actions[aid] = int(q_vals.argmax(dim=-1).item())

            next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
            reward = float(list(rewards.values())[0])
            done = any(terminations.values()) or any(truncations.values())
            next_obs_flat = np.concatenate([np.asarray(next_obs_dict[aid], dtype=np.float32) for aid in agent_ids])
            actions_arr = np.array([actions[aid] for aid in agent_ids], dtype=np.int64)

            buffer.push(obs_flat, actions_arr, reward, next_obs_flat, done)

            # Gradient step
            if buffer.size >= min_buffer_size:
                batch = buffer.sample(batch_size)
                loss = compute_qmix_loss(batch, agent_nets, target_agent_nets, mixer, target_mixer, gamma)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                optimizer.step()

            obs_dict = next_obs_dict
            if done:
                break

        env.close()

        # Hard-copy target networks
        if (episode + 1) % target_update_freq == 0:
            for net, tnet in zip(agent_nets, target_agent_nets):
                tnet.load_state_dict(net.state_dict())
            target_mixer.load_state_dict(mixer.state_dict())

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            es = eval_seed(seed) if seed is not None else None
            r = evaluate_qmix(agent_nets, agent_ids, scenario_name,
                              sim_seconds=eval_seconds, eval_sumo_seed=es)
            eval_history.append((episode + 1, r))

    # Save models (sanitize ids — real OSM junction names can contain ':' / '/')
    os.makedirs(save_dir, exist_ok=True)
    for aid, net in zip(agent_ids, agent_nets):
        sanitized_id = str(aid).replace(":", "_").replace("/", "_")
        torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_{scenario_name}_agent_{sanitized_id}.pth"))
    torch.save(mixer.state_dict(), os.path.join(save_dir, f"qmix_{scenario_name}_mixer.pth"))

    return eval_history


# =====================================================================
# Evaluation
# =====================================================================
def evaluate_qmix(agent_nets, agent_ids, scenario_name, sim_seconds=600, eval_sumo_seed=None):
    construct_seed = eval_sumo_seed if eval_sumo_seed is not None else "random"
    env = make_parallel_env(scenario_name, num_seconds=sim_seconds, sumo_seed=construct_seed)
    obs_dict, _ = env.reset(seed=eval_sumo_seed)
    total_reward = 0.0
    steps = 0

    while True:
        actions = {}
        with torch.no_grad():
            for i, aid in enumerate(agent_ids):
                obs_t = torch.FloatTensor(np.asarray(obs_dict[aid], dtype=np.float32)).unsqueeze(0)
                q_vals = agent_nets[i](obs_t)
                actions[aid] = int(q_vals.argmax(dim=-1).item())

        next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += float(list(rewards.values())[0])
        steps += 1
        obs_dict = next_obs_dict
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()
    print(f"  QMIX [{scenario_name}] eval reward: {total_reward:.2f} over {steps} steps")
    return total_reward


def run_qmix_scenario(scenario, num_episodes=200, eval_interval=20, seed=None, out_dir="outputs",
                      sim_seconds=1800, eval_seconds=600):
    """Train QMIX (CTDE) on any scenario (1x4 corridor or RESCO net), write a
    per-(scenario, seed) CSV of the eval history, and return it. Reused by
    run_seeded_experiment.py."""
    print(f"\n{'='*55}\nQMIX TRAINING: {scenario.upper()}  (seed={seed})\n{'='*55}")
    history = train_qmix(scenario_name=scenario, num_episodes=num_episodes,
                         eval_interval=eval_interval, seed=seed,
                         sim_seconds=sim_seconds, eval_seconds=eval_seconds)

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_seed{seed}" if seed is not None else ""
    out_csv = os.path.join(out_dir, f"qmix_results_{scenario}{suffix}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "episode", "eval_reward"])
        writer.writeheader()
        for episode, reward in history:
            writer.writerow({"scenario": scenario, "episode": episode, "eval_reward": reward})
    print(f"Saved {out_csv} ({len(history)} rows)")
    return history


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    log_fh = _open_log("ctde_training.log")
    current_phase = "startup"
    try:
        for scenario in SCENARIOS:
            current_phase = scenario
            run_qmix_scenario(scenario, num_episodes=200, eval_interval=20, seed=None)
    except BaseException:
        print(f"\n!!! CRASHED during phase: {current_phase} at {datetime.now().isoformat()} !!!")
        traceback.print_exc()
        raise
    finally:
        print(f"=== run_ctde_experiment.py finished at {datetime.now().isoformat()} ===")
        log_fh.close()