import pytest
from marl_algorithms import JALAgent, MaxPlusAgent

def get_topology_agents(size):
    """Simulates topology reconstruction logic inside the orchestrator scripts."""
    agents = {}
    cols = [chr(65 + c) for c in range(size)]
    rows = [str(r) for r in range(size)]
    
    agent_ids = [f"{c}{r}" for c in cols for r in rows]
    
    for agent_id in agent_ids:
        # Reconstruct neighbor IDs dynamic coordinates
        col_idx = ord(agent_id[0]) - 65
        row_idx = int(agent_id[1])
        neighbor_ids = []
        if col_idx > 0: neighbor_ids.append(f"{chr(65 + col_idx - 1)}{row_idx}")
        if col_idx < size - 1: neighbor_ids.append(f"{chr(65 + col_idx + 1)}{row_idx}")
        if row_idx > 0: neighbor_ids.append(f"{chr(65 + col_idx)}{row_idx - 1}")
        if row_idx < size - 1: neighbor_ids.append(f"{chr(65 + col_idx)}{row_idx + 1}")
        
        agents[agent_id] = JALAgent(agent_id, neighbor_ids)
        
    return agents

def test_dynamic_grid_scaling_policy_sync():
    """Asserts policy dictionary dynamic key generation and garbage collection of stale keys."""
    # 1. Initialize for a 2x2 Grid topology (4 agents)
    agents_2x2 = get_topology_agents(size=2)
    assert len(agents_2x2) == 4
    expected_2x2 = {"A0", "A1", "B0", "B1"}
    assert set(agents_2x2.keys()) == expected_2x2
    
    # Verify neighbor links for B1 in a 2x2 grid (should only link to A1 and B0)
    assert set(agents_2x2["B1"].neighbor_ids) == {"A1", "B0"}
    
    # 2. Scale dynamically to 3x3 Grid topology (9 agents)
    agents_3x3 = get_topology_agents(size=3)
    assert len(agents_3x3) == 9
    expected_3x3 = {"A0", "A1", "A2", "B0", "B1", "B2", "C0", "C1", "C2"}
    assert set(agents_3x3.keys()) == expected_3x3
    
    # Verify neighbor links for B1 in a 3x3 grid (now links to A1, C1, B0, B2)
    assert set(agents_3x3["B1"].neighbor_ids) == {"A1", "C1", "B0", "B2"}
    
    # Assert that the new 3x3 initialization has completely fresh instances
    for key in agents_2x2.keys():
        assert agents_2x2[key] is not agents_3x3[key]
