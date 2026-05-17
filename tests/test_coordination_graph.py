import numpy as np
import pytest
from marl_algorithms import MaxPlusAgent, JALAgent, run_max_plus_coordination

def test_jal_boundary_neighbor_history():
    """Asserts that JALAgent handles missing/empty neighbor actions without throwing KeyError."""
    # Initialize JALAgent at corner junction A0 (neighbors A1, B0)
    agent = JALAgent(agent_id="A0", neighbor_ids=["A1", "B0"])
    
    # 1. Standard JAL update with all neighbor actions present
    neighbor_actions = {"A1": 1, "B0": 0}
    next_neighbor_actions = {"A1": 0, "B0": 1}
    
    obs = np.array([2.0, 1.0])
    agent.update(
        obs, action=0, reward=-1.0, next_obs=obs,
        neighbor_actions=neighbor_actions,
        next_neighbor_actions=next_neighbor_actions,
        done=False
    )
    
    # 2. Boundary update with missing/non-existent neighbor node (e.g. B0 was deleted or inactive)
    missing_neighbor_actions = {"A1": 1} # B0 is missing!
    next_missing_neighbor_actions = {"A1": 0} # B0 is missing!
    
    # JAL should intercept B0's missing action gracefully and fall back to default action 0 without KeyError
    agent.update(
        obs, action=1, reward=-2.0, next_obs=obs,
        neighbor_actions=missing_neighbor_actions,
        next_neighbor_actions=next_missing_neighbor_actions,
        done=False
    )
    
    # The discretizer should map {"A1": 1} -> B0 defaults to 0 -> joint state index is 1
    assert agent.get_neighbor_action_state(missing_neighbor_actions) == 1

def test_max_plus_boundary_missing_communication():
    """Asserts that Max-Plus handles missing edge communication links gracefully and defaults to neutral values."""
    # A0 lists "A1" and "B0" as its neighbors
    agents_dict = {
        "A0": MaxPlusAgent(agent_id="A0", neighbor_ids=["A1", "B0"]),
        "A1": MaxPlusAgent(agent_id="A1", neighbor_ids=["A0"])
        # B0 is completely missing from the system config / not in agents_dict!
    }
    
    obs_dict = {
        "A0": np.array([3.0, 0.0]),
        "A1": np.array([0.0, 2.0]),
        "B0": np.array([1.0, 1.0]) # Even if present in observations, B0 has no policy network active
    }
    
    # Execution should finish successfully without raising KeyError when passing messages to/from B0
    actions = run_max_plus_coordination(agents_dict, obs_dict, iterations=3)
    
    assert "A0" in actions
    assert "A1" in actions
    assert "B0" not in actions # B0 is correctly excluded from coordination outputs since it has no agent policy
    
    # Validate actions are standard discrete phase values
    assert actions["A0"] in [0, 1]
    assert actions["A1"] in [0, 1]
