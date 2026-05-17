import os
import shutil
import xml.etree.ElementTree as ET
import pytest
import sumo_rl
from generate_bottleneck_network import build_bottleneck_scenario
from env_setup import QueueObservationFunction, custom_reward_fn

def test_scenario_bottleneck_generation():
    """Asserts that capacity drop bottleneck network is compiled and has exact 3-to-1 lane drop."""
    net_file = "bottleneck.net.xml"
    rou_file = "bottleneck.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the bottleneck scenario
        net, rou = build_bottleneck_scenario()
        
        assert net == net_file
        assert rou == rou_file
        assert os.path.exists(net_file)
        assert os.path.exists(rou_file)
        
        # Verify lane count drop via XML parsing of net file
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Check edges properties
        edges = root.findall("edge")
        bottleneck_edges = [e for e in edges if e.get("id") == "A_to_B"]
        assert len(bottleneck_edges) == 1, "Bottleneck edge A_to_B not found!"
        
        # Check lanes number on bottleneck edge
        lanes = bottleneck_edges[0].findall("lane")
        assert len(lanes) == 1, f"Expected exactly 1 lane for A_to_B bottleneck, got {len(lanes)}"
        
        # Verify West_to_A is a 3-lane edge
        incoming_edges = [e for e in edges if e.get("id") == "W_to_A"]
        assert len(incoming_edges) == 1
        assert len(incoming_edges[0].findall("lane")) == 3, "West incoming road must have 3 lanes!"
        
        # Test environment steps
        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=rou_file,
            use_gui=False,
            num_seconds=100,
            delta_time=5,
            reward_fn=custom_reward_fn,
            observation_class=QueueObservationFunction
        )
        obs, infos = env.reset()
        
        # Both A and B are junctions
        assert set(env.agents) == {"A", "B"}
        
        actions = {agent_id: 0 for agent_id in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(next_obs) == 2
        env.close()
        
    finally:
        # Clean up files
        for f in [net_file, rou_file]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
                
        # Restore backups
        try:
            if os.path.exists(net_backup):
                shutil.move(net_backup, net_file)
        except Exception:
            pass
            
        try:
            if os.path.exists(rou_backup):
                shutil.move(rou_backup, rou_file)
        except Exception:
            pass
