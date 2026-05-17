import os
import shutil
import xml.etree.ElementTree as ET
import pytest
import sumo_rl
from generate_spillback_network import build_spillback_scenario
from env_setup import QueueObservationFunction, custom_reward_fn

def test_scenario_spillback_generation():
    """Asserts that 50m grid network compiles correctly and steps without observations errors."""
    net_file = "grid_2x2_spillback.net.xml"
    rou_file = "grid_2x2_spillback.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the spillback scenario
        net, rou = build_spillback_scenario(size=2)
        
        assert net == net_file
        assert rou == rou_file
        assert os.path.exists(net_file)
        assert os.path.exists(rou_file)
        
        # Verify 50m length of segments inside net file
        tree = ET.parse(net_file)
        root = tree.getroot()
        
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
        
        # Ensure 2x2 grid has 4 agents (A0, A1, B0, B1)
        assert len(env.agents) == 4
        
        actions = {agent_id: 0 for agent_id in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(next_obs) == 4
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
