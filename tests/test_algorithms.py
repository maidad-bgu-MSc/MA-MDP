import os
import numpy as np
import torch
import pytest
from marl_algorithms import TabularQLearningAgent, QMIXAgentNetwork

def test_tabular_q_agent():
    """Verifies TabularQLearningAgent state discretization, action selection, and learning update."""
    agent = TabularQLearningAgent(agent_id="A0", num_states=25)
    # The new obs from QueueObservationFunction is exactly [discrete_local, discrete_upstream]
    # where each value is an integer 0-4.
    obs = np.array([3.0, 0.0])
    
    action = agent.compute_action(obs, explore=True)
    assert action in [0, 1]
    
    initial_q_val = agent.q_table.copy()
    agent.update(obs, action, -2.5, obs, done=False)
    # The Q-table should have been successfully updated (non-identical due to step)
    assert not np.array_equal(initial_q_val, agent.q_table), "Q-table was not updated!"

def test_qmix_network():
    """Verifies QMIXAgentNetwork PyTorch forward pass output shape and logit bounds."""
    agent = QMIXAgentNetwork(obs_dim=2, action_dim=2)
    obs_tensor = torch.FloatTensor([[1.5, 0.5]])
    
    with torch.no_grad():
        q_values = agent(obs_tensor)
        
    assert q_values.shape == (1, 2), f"Expected Q-value shape (1, 2), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaNs!"
