import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# State Discretization Wrapper
# =====================================================================
def discretize_queue(q_length):
    """5-Bin Discretization for queue lengths:
    0: 0 cars
    1: 1-5 cars
    2: 6-15 cars
    3: 16-29 cars
    4: 30+ cars
    """
    if q_length <= 0:
        return 0
    elif q_length <= 5:
        return 1
    elif q_length <= 15:
        return 2
    elif q_length <= 29:
        return 3
    else:
        return 4

def get_discrete_state(obs):
    """Converts continuous lane-queue observation into a discrete state index (0 to 624)."""
    # Assuming obs is already discretized from QueueObservationFunction into [bin1, bin2, bin3, bin4]
    # We map this 5x5x5x5 space to a single integer 0-624
    if len(obs) == 4:
        return int(obs[0]) * 125 + int(obs[1]) * 25 + int(obs[2]) * 5 + int(obs[3])
    elif len(obs) == 2:
        return int(obs[0]) * 5 + int(obs[1])
    else:
        raise ValueError(f"Unsupported observation shape: {obs.shape}")

# =====================================================================
# 1. Independent Tabular Q-Learning (Decentralized Classical)
# =====================================================================
class TabularQLearningAgent:
    """Decentralized tabular Q-Learning with discretized state space."""
    def __init__(self, agent_id, num_states=625, num_actions=2, lr=0.01, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def compute_action(self, obs, explore=True):
        state = get_discrete_state(obs)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = get_discrete_state(obs)
        next_state = get_discrete_state(next_obs)
        best_next_action = np.argmax(self.q_table[next_state])
        
        td_target = reward + (0.0 if done else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

class HystereticQLearningAgent:
    """Decentralized Hysteretic Q-Learning for multi-agent coordination."""
    def __init__(self, agent_id, num_states=625, num_actions=2, alpha=0.01, beta=0.001, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Optimistic learning rate for positive TD error
        self.beta = beta    # Pessimistic learning rate for negative TD error
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))

    def compute_action(self, obs, explore=True):
        state = get_discrete_state(obs)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = get_discrete_state(obs)
        next_state = get_discrete_state(next_obs)
        best_next_action = np.argmax(self.q_table[next_state])
        
        td_target = reward + (0.0 if done else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        
        # Hysteretic update
        if td_error >= 0:
            self.q_table[state, action] += self.alpha * td_error
        else:
            self.q_table[state, action] += self.beta * td_error


# =====================================================================
# 2. Tabular VDN (Value Decomposition Networks)
# =====================================================================
class TabularVDNAgents:
    """Cooperative tabular VDN: Q_total = sum of individual Q_i(s_i, a_i).

    Centralized training: all agents share a single TD error computed on
    Q_total. Decentralized execution: each agent acts greedily on its own
    Q-table independently.
    """
    def __init__(self, agent_ids, num_states=625, num_actions=2,
                 lr=0.01, gamma=0.95, epsilon=0.1):
        self.agent_ids = list(agent_ids)
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_tables = {aid: np.zeros((num_states, num_actions)) for aid in self.agent_ids}

    def compute_action(self, agent_id, obs, explore=True):
        state = get_discrete_state(obs)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_tables[agent_id][state]))

    def update(self, obs_dict, action_dict, reward, next_obs_dict, done):
        """Centralized VDN update: same TD scalar applied to every agent's Q-table entry."""
        q_total = 0.0
        q_next_total = 0.0
        states = {}
        for aid in self.agent_ids:
            s = get_discrete_state(obs_dict[aid])
            s_next = get_discrete_state(next_obs_dict[aid])
            states[aid] = s
            q_total += self.q_tables[aid][s, action_dict[aid]]
            q_next_total += np.max(self.q_tables[aid][s_next])

        td_target = reward + (0.0 if done else self.gamma * q_next_total)
        td_error = td_target - q_total
        delta = self.lr * td_error
        for aid in self.agent_ids:
            self.q_tables[aid][states[aid], action_dict[aid]] += delta


# =====================================================================
# 3. QMIX Centralized Mixing Deep Network
# =====================================================================
class QMIXMixingNetwork(nn.Module):
    """Centralized monotonic utility mixing network using PyTorch hypernetworks."""
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super(QMIXMixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim
        
        # Hypernetwork for weight matrix W1: maps state to positive weight vector
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents * mixing_embed_dim)
        )
        
        # Hypernetwork for weight matrix W2: maps state to positive weight vector
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mixing_embed_dim * 1)
        )
        
        # Biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs: [batch_size, num_agents]
        # states: [batch_size, state_dim]
        bs = agent_qs.size(0)
        
        # Absolute weights guarantee monotonicity
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.embed_dim)
        
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        
        # Mix agent values
        agent_qs = agent_qs.view(bs, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(bs, 1)

class QMIXAgentNetwork(nn.Module):
    """Individual deep Q-network for QMIX execution."""
    def __init__(self, obs_dim=4, action_dim=2):
        super(QMIXAgentNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs):
        return self.fc(obs)
