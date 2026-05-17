import os
import numpy as np
import torch
import pytest
from marl_algorithms import (
    FixedTimeController, TabularQLearningAgent, SARSAAgent,
    WLearningAgent, JALAgent, MaxPlusAgent, QMIXAgentNetwork,
    run_max_plus_coordination
)

def test_fixed_time_agent():
    """Verifies FixedTimeController action selection and update interface."""
    agent = FixedTimeController(agent_id="A0")
    obs = np.array([2.0, 1.0])
    
    # Cycles every 30s (6 steps), select action 0 initially
    action = agent.compute_action(obs, explore=False)
    assert action in [0, 1], f"Invalid action: {action}"
    
    # Should not raise exception on dummy updates
    agent.update(obs, action, -1.0, obs, done=False)

def test_tabular_q_agent():
    """Verifies TabularQLearningAgent state discretization, action selection, and learning update."""
    agent = TabularQLearningAgent(agent_id="A0")
    obs = np.array([3.0, 0.0])
    
    action = agent.compute_action(obs, explore=True)
    assert action in [0, 1]
    
    initial_q_val = agent.q_table.copy()
    agent.update(obs, action, -2.5, obs, done=False)
    # The Q-table should have been successfully updated (non-identical due to step)
    assert not np.array_equal(initial_q_val, agent.q_table), "Q-table was not updated!"

def test_sarsa_agent():
    """Verifies SARSAAgent action selection and on-policy learning update."""
    agent = SARSAAgent(agent_id="A0")
    obs = np.array([0.0, 5.0])
    
    action = agent.compute_action(obs, explore=True)
    next_action = agent.compute_action(obs, explore=True)
    
    initial_q_val = agent.q_table.copy()
    agent.update(obs, action, -1.0, obs, next_action, done=False)
    assert not np.array_equal(initial_q_val, agent.q_table), "Q-table was not updated!"

def test_w_learning_agent():
    """Verifies Distributed W-Learning competitor agent state queues and updates."""
    agent = WLearningAgent(agent_id="A0")
    obs = np.array([4.0, 2.0])
    
    action = agent.compute_action(obs, explore=True)
    assert action in [0, 1]
    
    initial_q_val = agent.q_tables[0].copy()
    agent.update(obs, action, -3.0, obs, done=False)
    assert not np.array_equal(initial_q_val, agent.q_tables[0]), "Distributed Q-table was not updated!"

def test_jal_agent():
    """Verifies Joint-Action Learner neighbor action-history mapping and updates."""
    neighbor_ids = ["A1", "B0"]
    agent = JALAgent(agent_id="A0", neighbor_ids=neighbor_ids)
    obs = np.array([1.0, 1.0])
    
    action = agent.compute_action(obs, explore=True)
    assert action in [0, 1]
    
    neighbor_actions = {"A1": 0, "B0": 1}
    next_neighbor_actions = {"A1": 1, "B0": 0}
    
    initial_q_val = agent.q_table.copy()
    agent.update(
        obs, action, -1.5, obs,
        neighbor_actions, next_neighbor_actions,
        done=False
    )
    assert not np.array_equal(initial_q_val, agent.q_table), "Joint Q-table was not updated!"

def test_max_plus_agent():
    """Verifies Max-Plus agent edge utilities and coordination message passing."""
    neighbor_ids = ["A1", "B0"]
    agents = {
        "A0": MaxPlusAgent(agent_id="A0", neighbor_ids=["A1", "B0"]),
        "A1": MaxPlusAgent(agent_id="A1", neighbor_ids=["A0"]),
        "B0": MaxPlusAgent(agent_id="B0", neighbor_ids=["A0"])
    }
    
    obs_dict = {
        "A0": np.array([2.0, 0.0]),
        "A1": np.array([0.0, 4.0]),
        "B0": np.array([3.0, 3.0])
    }
    
    # 1. Test coordination message passing
    actions = run_max_plus_coordination(agents, obs_dict)
    assert "A0" in actions
    assert actions["A0"] in [0, 1]
    
    # 2. Test learning update
    agent_a0 = agents["A0"]
    initial_q_local = agent_a0.q_local.copy()
    
    neighbor_obs_dict = {
        "A1": obs_dict["A1"],
        "B0": obs_dict["B0"]
    }
    neighbor_actions_dict = {
        "A1": 0,
        "B0": 1
    }
    
    agent_a0.update(
        obs_dict["A0"], actions["A0"], -1.0, obs_dict["A0"], False,
        neighbor_obs_dict, neighbor_actions_dict
    )
    assert not np.array_equal(initial_q_local, agent_a0.q_local), "Local Max-Plus Q-table was not updated!"

def test_qmix_network():
    """Verifies QMIXAgentNetwork PyTorch forward pass output shape and logit bounds."""
    agent = QMIXAgentNetwork(obs_dim=2, action_dim=2)
    obs_tensor = torch.FloatTensor([[1.5, 0.5]])
    
    with torch.no_grad():
        q_values = agent(obs_tensor)
        
    assert q_values.shape == (1, 2), f"Expected Q-value shape (1, 2), got {q_values.shape}"
    assert not torch.isnan(q_values).any(), "Q-values contain NaNs!"
