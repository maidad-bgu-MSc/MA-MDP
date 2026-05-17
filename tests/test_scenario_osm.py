import os
import shutil
import xml.etree.ElementTree as ET
import pytest
import sumo_rl
from generate_osm_network import build_osm_scenario
from env_setup import QueueObservationFunction, custom_reward_fn

def test_scenario_osm_generation():
    """Asserts that the custom asymmetric Beer Sheva corridor compiles cleanly and maps varied junctions."""
    net_file = "osm.net.xml"
    rou_file = "osm.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the OSM asymmetric scenario
        net, rou = build_osm_scenario()
        
        assert net == net_file
        assert rou == rou_file
        assert os.path.exists(net_file)
        assert os.path.exists(rou_file)
        
        # Verify T-junctions (A, C) vs 4-way (B) inside net file
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Check nodes are registered
        junctions = root.findall("junction")
        junction_ids = [j.get("id") for j in junctions]
        assert "A" in junction_ids
        assert "B" in junction_ids
        assert "C" in junction_ids
        
        # Test environment execution on asymmetric lanes
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
        
        # Validate exact agent registrations
        assert len(env.agents) == 3, f"Expected 3 agents (A, B, C), got {len(env.agents)}"
        assert set(env.agents) == {"A", "B", "C"}
        
        # Step env
        actions = {agent_id: 0 for agent_id in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(next_obs) == 3
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
