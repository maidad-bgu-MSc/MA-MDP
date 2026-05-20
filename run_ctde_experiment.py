import os
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from marl_algorithms import QMIXAgentNetwork, QMIXMixingNetwork
from simulator.problem_generator import SCENARIOS, make_problem_parallel_env

NUM_AGENTS = 4
OBS_DIM = 4
STATE_DIM = NUM_AGENTS * OBS_DIM  # 16 — concatenated agent observations fed to mixer


# =====================================================================
# Replay Buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self._obs = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self._actions = np.zeros((capacity, NUM_AGENTS), dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, STATE_DIM), dtype=np.float32)
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

    # Split flat obs into per-agent observations
    obs_per_agent = obs_flat.view(B, NUM_AGENTS, OBS_DIM)
    next_obs_per_agent = next_obs_flat.view(B, NUM_AGENTS, OBS_DIM)

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
):
    agent_nets = [QMIXAgentNetwork(obs_dim=OBS_DIM, action_dim=2) for _ in range(NUM_AGENTS)]
    target_agent_nets = [copy.deepcopy(net) for net in agent_nets]
    mixer = QMIXMixingNetwork(num_agents=NUM_AGENTS, state_dim=STATE_DIM)
    target_mixer = copy.deepcopy(mixer)

    all_params = list(mixer.parameters())
    for net in agent_nets:
        all_params += list(net.parameters())
    optimizer = optim.Adam(all_params, lr=lr)

    buffer = ReplayBuffer(capacity=buffer_capacity)
    eval_history = []

    for episode in tqdm(range(num_episodes), desc=f"QMIX [{scenario_name}]"):
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * episode / epsilon_decay_episodes)

        env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds)
        agent_ids = env.possible_agents
        obs_dict, _ = env.reset()

        while True:
            obs_flat = np.concatenate([obs_dict[aid].astype(np.float32) for aid in agent_ids])

            # Epsilon-greedy action selection
            actions = {}
            with torch.no_grad():
                for i, aid in enumerate(agent_ids):
                    if np.random.rand() < epsilon:
                        actions[aid] = np.random.randint(2)
                    else:
                        obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
                        q_vals = agent_nets[i](obs_t)
                        actions[aid] = int(q_vals.argmax(dim=-1).item())

            next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
            reward = float(list(rewards.values())[0])
            done = any(terminations.values()) or any(truncations.values())
            next_obs_flat = np.concatenate([next_obs_dict[aid].astype(np.float32) for aid in agent_ids])
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
            r = evaluate_qmix(agent_nets, scenario_name, sim_seconds=eval_seconds)
            eval_history.append((episode + 1, r))

    # Save models
    os.makedirs(save_dir, exist_ok=True)
    for i, (aid, net) in enumerate(zip(agent_ids, agent_nets)):
        torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_{scenario_name}_agent_{aid}.pth"))
    torch.save(mixer.state_dict(), os.path.join(save_dir, f"qmix_{scenario_name}_mixer.pth"))

    return eval_history


# =====================================================================
# Evaluation
# =====================================================================
def evaluate_qmix(agent_nets, scenario_name, sim_seconds=600):
    env = make_problem_parallel_env(scenario_name, num_seconds=sim_seconds)
    agent_ids = env.possible_agents
    obs_dict, _ = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        actions = {}
        with torch.no_grad():
            for i, aid in enumerate(agent_ids):
                obs_t = torch.FloatTensor(obs_dict[aid]).unsqueeze(0)
                q_vals = agent_nets[i](obs_t)
                actions[aid] = int(q_vals.argmax(dim=-1).item())

        next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        total_reward += float(list(rewards.values())[0])
        steps += 1
        obs_dict = next_obs_dict
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()
    mean_reward = total_reward / max(steps, 1)
    print(f"  QMIX [{scenario_name}] eval reward: {total_reward:.2f} over {steps} steps")
    return total_reward


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    results = []

    for scenario in SCENARIOS:
        print(f"\n{'='*55}")
        print(f"QMIX TRAINING: {scenario.upper()}")
        print(f"{'='*55}")
        history = train_qmix(scenario_name=scenario, num_episodes=200, eval_interval=20)
        for episode, reward in history:
            results.append({"scenario": scenario, "episode": episode, "eval_reward": reward})

    out_csv = os.path.join("outputs", "qmix_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "episode", "eval_reward"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nQMIX results saved to {out_csv}")