import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# State Discretization Wrapper
# =====================================================================
def discretize_queue(q_length):
    """Groups halted vehicle counts (queues) into 4 discrete bins."""
    if q_length <= 0:
        return 0
    elif q_length <= 3:
        return 1
    elif q_length <= 7:
        return 2
    else:
        return 3

def get_discrete_state(obs):
    """Converts continuous lane-queue observation into a discrete state index (0 to 15)."""
    # In sumo-rl, observation lists the halted vehicle count per lane
    q1 = discretize_queue(obs[0])
    q2 = discretize_queue(obs[1])
    return q1 * 4 + q2

# =====================================================================
# 1. Fixed-Time Controller (Non-MDP Baseline)
# =====================================================================
class FixedTimeController:
    """Cycle traffic signals in a standard round-robin fashion."""
    def __init__(self, agent_id, cycle_steps=6):
        self.agent_id = agent_id
        self.cycle_steps = cycle_steps
        self.step_count = 0
        self.current_action = 0

    def compute_action(self, obs, explore=False):
        # Alternate phases every cycle_steps steps
        if self.step_count > 0 and self.step_count % self.cycle_steps == 0:
            self.current_action = 1 - self.current_action
        self.step_count += 1
        return self.current_action

    def update(self, *args, **kwargs):
        pass  # Non-learning model

# =====================================================================
# 2. Independent Tabular Q-Learning (Decentralized Classical)
# =====================================================================
class TabularQLearningAgent:
    """Decentralized tabular Q-Learning with discretized state space."""
    def __init__(self, agent_id, num_states=16, num_actions=2, lr=0.1, gamma=0.95, epsilon=0.1):
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

# =====================================================================
# 3. Independent SARSA Agent
# =====================================================================
class SARSAAgent:
    """Classical decentralized on-policy SARSA control."""
    def __init__(self, agent_id, num_states=16, num_actions=2, lr=0.1, gamma=0.95, epsilon=0.1):
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

    def update(self, obs, action, reward, next_obs, next_action, done):
        state = get_discrete_state(obs)
        next_state = get_discrete_state(next_obs)
        
        td_target = reward + (0.0 if done else self.gamma * self.q_table[next_state, next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

# =====================================================================
# 4. Distributed W-Learning Agent
# =====================================================================
class WLearningAgent:
    """Learns resource allocation priorities across competing incoming queues."""
    def __init__(self, agent_id, num_lanes=2, num_states_per_lane=4, num_actions=2, lr=0.1, beta=0.05, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.num_lanes = num_lanes
        self.num_states_per_lane = num_states_per_lane
        self.num_actions = num_actions
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Sub-Q-tables for each lane (competing directions)
        self.q_tables = [np.zeros((num_states_per_lane, num_actions)) for _ in range(num_lanes)]
        # W-tables representing how much each lane 'cares' about winning the action slot
        self.w_tables = [np.zeros(num_states_per_lane) for _ in range(num_lanes)]

    def compute_action(self, obs, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
            
        discrete_lanes = [discretize_queue(obs[i]) for i in range(self.num_lanes)]
        
        # Each lane nominates its preferred action
        nominated_actions = []
        for i in range(self.num_lanes):
            nominated_actions.append(np.argmax(self.q_tables[i][discrete_lanes[i]]))
            
        # Select action of the lane with the highest W-value (regret)
        w_values = [self.w_tables[i][discrete_lanes[i]] for i in range(self.num_lanes)]
        winner_lane = np.argmax(w_values)
        return int(nominated_actions[winner_lane])

    def update(self, obs, action, reward, next_obs, done):
        discrete_lanes = [discretize_queue(obs[i]) for i in range(self.num_lanes)]
        next_discrete_lanes = [discretize_queue(next_obs[i]) for i in range(self.num_lanes)]
        
        w_values = [self.w_tables[i][discrete_lanes[i]] for i in range(self.num_lanes)]
        winner_lane = np.argmax(w_values)
        
        for i in range(self.num_lanes):
            s_i = discrete_lanes[i]
            s_next_i = next_discrete_lanes[i]
            
            # Local reward approximation based on waiting queues
            lane_reward = -next_obs[i]
            
            if i == winner_lane:
                best_next_action = np.argmax(self.q_tables[i][s_next_i])
                target = lane_reward + (0.0 if done else self.gamma * self.q_tables[i][s_next_i, best_next_action])
                self.q_tables[i][s_i, action] += self.lr * (target - self.q_tables[i][s_i, action])
            else:
                pref_action = np.argmax(self.q_tables[i][s_i])
                regret = self.q_tables[i][s_i, pref_action] - self.q_tables[i][s_i, action]
                self.w_tables[i][s_i] += self.beta * (regret - self.w_tables[i][s_i])

# =====================================================================
# 5. Joint-Action Learners (JAL) / Coordinated Q-Learning
# =====================================================================
class JALAgent:
    """Observes adjacent physical neighbors' action history to coordinate action selection."""
    def __init__(self, agent_id, neighbor_ids, num_states=16, num_actions=2, lr=0.1, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.neighbor_ids = neighbor_ids
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.num_neighbor_actions = 2 ** len(neighbor_ids) if len(neighbor_ids) > 0 else 1
        # Joint Q-Table: Q(local_state, action_self, joint_neighbor_action)
        self.q_table = np.zeros((num_states, num_actions, self.num_neighbor_actions))
        self.prev_neighbor_actions = 0

    def get_neighbor_action_state(self, neighbor_actions_dict):
        state = 0
        for idx, n_id in enumerate(self.neighbor_ids):
            a = neighbor_actions_dict.get(n_id, 0)
            state += a * (2 ** idx)
        return state

    def compute_action(self, obs, explore=True):
        state = get_discrete_state(obs)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return int(np.argmax(self.q_table[state, :, self.prev_neighbor_actions]))

    def update(self, obs, action, reward, next_obs, neighbor_actions, next_neighbor_actions, done):
        state = get_discrete_state(obs)
        next_state = get_discrete_state(next_obs)
        
        curr_n_state = self.get_neighbor_action_state(neighbor_actions)
        next_n_state = self.get_neighbor_action_state(next_neighbor_actions)
        
        best_next_action = np.argmax(self.q_table[next_state, :, next_n_state])
        td_target = reward + (0.0 if done else self.gamma * self.q_table[next_state, best_next_action, next_n_state])
        td_error = td_target - self.q_table[state, action, curr_n_state]
        self.q_table[state, action, curr_n_state] += self.lr * td_error
        
        self.prev_neighbor_actions = next_n_state

# =====================================================================
# 6. Max-Plus Algorithm (Coordination Graphs)
# =====================================================================
class MaxPlusAgent:
    """Uses pairwise edge utilities and local payoff networks for coordination."""
    def __init__(self, agent_id, neighbor_ids, num_states=16, num_actions=2, lr=0.1, gamma=0.95, epsilon=0.1):
        self.agent_id = agent_id
        self.neighbor_ids = neighbor_ids
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Local Q Table: Q_i(s_i, a_i)
        self.q_local = np.zeros((num_states, num_actions))
        
        # Pairwise Q Tables: Q_{ij}(s_i, s_j, a_i, a_j)
        self.q_pairs = {n_id: np.zeros((num_states, num_states, num_actions, num_actions)) for n_id in neighbor_ids}

    def get_local_payoff(self, obs, action):
        state = get_discrete_state(obs)
        return self.q_local[state, action]

    def get_pair_payoff(self, neighbor_id, obs, neighbor_obs, action, neighbor_action):
        state_self = get_discrete_state(obs)
        state_neigh = get_discrete_state(neighbor_obs)
        return self.q_pairs[neighbor_id][state_self, state_neigh, action, neighbor_action]

    def update(self, obs, action, reward, next_obs, done, neighbor_obs_dict, neighbor_actions_dict):
        state = get_discrete_state(obs)
        next_state = get_discrete_state(next_obs)
        best_next_action = np.argmax(self.q_local[next_state])
        
        target = reward + (0.0 if done else self.gamma * self.q_local[next_state, best_next_action])
        self.q_local[state, action] += self.lr * (target - self.q_local[state, action])
        
        # Update pairwise edges
        for n_id in self.neighbor_ids:
            if n_id in neighbor_obs_dict and n_id in neighbor_actions_dict:
                s_neigh = get_discrete_state(neighbor_obs_dict[n_id])
                a_neigh = neighbor_actions_dict[n_id]
                
                # Coordinated pairwise utility update
                self.q_pairs[n_id][state, s_neigh, action, a_neigh] += self.lr * (
                    reward - self.q_pairs[n_id][state, s_neigh, action, a_neigh]
                )

def run_max_plus_coordination(agents_dict, obs_dict, iterations=4):
    """Executes message-passing coordination over the physical grid graphs."""
    messages = {}
    
    # Initialize messages
    for agent_id, agent in agents_dict.items():
        for n_id in agent.neighbor_ids:
            messages[(agent_id, n_id)] = np.zeros(2)
            
    # Passing messages iteratively
    for _ in range(iterations):
        new_messages = {}
        for (sender, receiver) in messages.keys():
            agent_sender = agents_dict[sender]
            obs_sender = obs_dict[sender]
            obs_receiver = obs_dict[receiver]
            
            msg = np.zeros(2)
            for a_recv in range(2):
                terms = []
                for a_send in range(2):
                    local_pay = agent_sender.get_local_payoff(obs_sender, a_send)
                    pair_pay = agent_sender.get_pair_payoff(receiver, obs_sender, obs_receiver, a_send, a_recv)
                    
                    incoming_sum = sum(
                        messages.get((n_id, sender), np.zeros(2))[a_send]
                        for n_id in agent_sender.neighbor_ids if n_id != receiver
                    )
                    terms.append(local_pay + pair_pay + incoming_sum)
                msg[a_recv] = max(terms)
                
            # Normalization to keep numerical stability
            msg -= np.mean(msg)
            new_messages[(sender, receiver)] = msg
        messages = new_messages
        
    # Act coordinately
    coordinated_actions = {}
    for agent_id, agent in agents_dict.items():
        obs_self = obs_dict[agent_id]
        total_utility = np.zeros(2)
        for a in range(2):
            local_val = agent.get_local_payoff(obs_self, a)
            msg_sum = sum(messages.get((n_id, agent_id), np.zeros(2))[a] for n_id in agent.neighbor_ids)
            total_utility[a] = local_val + msg_sum
        coordinated_actions[agent_id] = int(np.argmax(total_utility))
        
    return coordinated_actions

# =====================================================================
# 7. QMIX Centralized Mixing Deep Network
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
    def __init__(self, obs_dim=2, action_dim=2):
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
