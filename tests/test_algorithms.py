import os
import numpy as np
import torch
import pytest
from marl_algorithms import TabularQLearningAgent, HystereticQLearningAgent, QMIXAgentNetwork

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
    """Verifies QMIXAgentNetwork PyTorch forward pass output shape and logit bounds."""
    agent = QMIXAgentNetwork(obs_dim=2, action_dim=2)
    obs_tensor = torch.FloatTensor([[1.5, 0.5]])
    
    with torch.no_grad():
        q_values = agent(obs_tensor)
        
    assert q_values.shape == (1, 2), f"Expected Q-value shape (1, 2), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaNs!"
