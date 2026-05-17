import numpy as np
import pytest
from marl_algorithms import get_discrete_state, discretize_queue

def test_discretize_queue_boundary_values():
    """Asserts that discretize_queue maps negative, float, and massive queues correctly."""
    # Negative queue -> should map safely to 0
    assert discretize_queue(-1) == 0
    assert discretize_queue(-10.5) == 0
    assert discretize_queue(0) == 0
    
    # Standard bounds
    assert discretize_queue(1) == 1
    assert discretize_queue(3) == 1
    assert discretize_queue(3.5) == 2
    assert discretize_queue(7) == 2
    
    # Extreme queues -> should cleanly clamp to 3
    assert discretize_queue(8) == 3
    assert discretize_queue(100) == 3
    assert discretize_queue(9999.9) == 3

def test_get_discrete_state_integrity():
    """Asserts that joint-state discretization safely outputs values in [0, 15] boundaries."""
    # Test extreme positive boundary
    obs_max = np.array([500.0, 1000.0])
    assert get_discrete_state(obs_max) == 15 # 3 * 4 + 3 = 15
    
    # Test extreme negative boundary
    obs_min = np.array([-50.0, -100.0])
    assert get_discrete_state(obs_min) == 0 # 0 * 4 + 0 = 0
    
    # Test float values
    obs_floats = np.array([2.5, 4.8])
    assert get_discrete_state(obs_floats) == 6 # 1 * 4 + 2 = 6

def test_obs_adaptation_handling():
    """Asserts that dynamic slicing/padding prevents index errors for non-conforming lengths."""
    from watch_agents import adapt_obs_dict
    
    # 1. Under-dimensioned state (e.g. corner junction with 1 queue instead of 2)
    bad_obs_dict_1 = {"corner_agent": np.array([3.0])}
    adapted_1 = adapt_obs_dict(bad_obs_dict_1)
    
    # Should pad with constant 0
    assert len(adapted_1["corner_agent"]) == 2
    assert adapted_1["corner_agent"][0] == 3.0
    assert adapted_1["corner_agent"][1] == 0.0
    assert get_discrete_state(adapted_1["corner_agent"]) == 4 # 1 * 4 + 0 = 4
    
    # 2. Over-dimensioned state (e.g. center junction with 4 queues instead of 2)
    bad_obs_dict_2 = {"center_agent": np.array([1.0, 5.0, 8.0, 12.0])}
    adapted_2 = adapt_obs_dict(bad_obs_dict_2)
    
    # Should slice to first 2 elements
    assert len(adapted_2["center_agent"]) == 2
    assert adapted_2["center_agent"][0] == 1.0
    assert adapted_2["center_agent"][1] == 5.0
    assert get_discrete_state(adapted_2["center_agent"]) == 6 # 1 * 4 + 2 = 6
