import os
import numpy as np
import torch
import pytest
from marl_algorithms import (TabularQLearningAgent, HystereticQLearningAgent,
                              TabularVDNAgents, QMIXAgentNetwork, QMIXMixingNetwork)

def test_tabular_q_agent():
    """Verifies TabularQLearningAgent state discretization, action selection, and learning update."""
    agent = TabularQLearningAgent(agent_id="A0", num_states=625)
    # The new obs from QueueObservationFunction is a 4D array [local_ew, local_ns, other_ew, other_ns]
    obs = np.array([3.0, 0.0, 1.0, 2.0])
    
    action = agent.compute_action(obs, explore=True)
    assert action in [0, 1]
    
    initial_q_val = agent.q_table.copy()
    agent.update(obs, action, -2.5, obs, done=False)
    assert not np.array_equal(initial_q_val, agent.q_table), "Q-table was not updated!"

def test_hysteretic_q_agent():
    """Verifies HystereticQLearningAgent applies different learning rates for positive and negative TD errors."""
    agent = HystereticQLearningAgent(agent_id="A0", num_states=625, alpha=0.5, beta=0.1)
    obs = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Force a positive TD error
    initial_q = agent.q_table.copy()
    agent.update(obs, action=0, reward=10.0, next_obs=obs, done=True)
    positive_update_amount = agent.q_table[0, 0] - initial_q[0, 0]
    
    # Force a negative TD error
    initial_q = agent.q_table.copy()
    agent.update(obs, action=0, reward=-100.0, next_obs=obs, done=True)
    negative_update_amount = agent.q_table[0, 0] - initial_q[0, 0]
    
    # The positive update should be scaled by alpha (0.5), negative by beta (0.1)
    # Alpha is 5x larger than beta, so the magnitude of update per unit error should be different.
    assert positive_update_amount > 0
    assert negative_update_amount < 0
    # positive error = 10, update = 10 * 0.5 = 5
    # negative error = -100 - 5 = -105, update = -105 * 0.1 = -10.5
    assert positive_update_amount == 5.0

def test_qmix_network():
    """Verifies QMIXAgentNetwork forward pass output shape with 4D obs."""
    agent = QMIXAgentNetwork(obs_dim=4, action_dim=2)
    obs_tensor = torch.FloatTensor([[1.5, 0.5, 2.0, 1.0]])

    with torch.no_grad():
        q_values = agent(obs_tensor)

    assert q_values.shape == (1, 2), f"Expected Q-value shape (1, 2), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaNs!"


def test_qmix_mixing_network():
    """Verifies QMIXMixingNetwork output shape and no NaNs with state_dim=16."""
    mixer = QMIXMixingNetwork(num_agents=4, state_dim=16, mixing_embed_dim=32)
    agent_qs = torch.FloatTensor([[1.0, -0.5, 2.0, -1.0]])  # (B=1, 4)
    state = torch.FloatTensor([[0.0] * 16])                  # (B=1, 16)

    with torch.no_grad():
        q_tot = mixer(agent_qs, state)

    assert q_tot.shape == (1, 1), f"Expected (1, 1), got {q_tot.shape}"
    assert not torch.isnan(q_tot).any(), "Q_tot contains NaNs!"


def test_tabular_vdn_update():
    """Verifies TabularVDNAgents centralized update applies equal TD delta to all agents."""
    agent_ids = ["A0", "B0", "C0", "D0"]
    # gamma=0 → td_target = reward; lr=0.5 → delta = 0.5 * reward
    vdn = TabularVDNAgents(agent_ids=agent_ids, num_states=625, lr=0.5, gamma=0.0, epsilon=0.0)

    obs = {aid: np.array([0.0, 0.0, 0.0, 0.0]) for aid in agent_ids}
    actions = {aid: 0 for aid in agent_ids}

    # All Q-tables start at 0; q_total=0, q_next=0, td_error = 4.0 - 0 = 4.0, delta = 0.5*4.0 = 2.0
    vdn.update(obs, actions, reward=4.0, next_obs_dict=obs, done=True)

    for aid in agent_ids:
        assert abs(vdn.q_tables[aid][0, 0] - 2.0) < 1e-9, \
            f"VDN Q[{aid}][0,0] = {vdn.q_tables[aid][0,0]}, expected 2.0"
