import os
import shutil
import xml.etree.ElementTree as ET
import pytest
import sumo_rl
from generate_platoon_network import build_platoon_scenario
from env_setup import QueueObservationFunction, custom_reward_fn

def test_scenario_platoon_generation():
    """Asserts that platoon network and route XML files are compiled, and structures match specifications."""
    net_file = "grid_2x2_platoon.net.xml"
    rou_file = "grid_2x2_platoon.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the platoon scenario
        net, rou = build_platoon_scenario(size=2)
        
        # Verify physical file creations
        assert net == net_file
        assert rou == rou_file
        assert os.path.exists(net_file)
        assert os.path.exists(rou_file)
        
        # Parse XML route file to assert platoon structures
        tree = ET.parse(rou_file)
        root = tree.getroot()
        
        vehicles = root.findall("vehicle")
        flows = root.findall("flow")
        
        # Check platoon injections
        platoon_vehicles = [v for v in vehicles if "platoon_veh" in v.get("id")]
        assert len(platoon_vehicles) > 0, "No platoon vehicle bursts generated!"
        assert len(platoon_vehicles) == 480, f"Expected 480 platoon vehicles, got {len(platoon_vehicles)}"
        
        # Check steady cross traffic flows
        assert len(flows) > 0, "No cross-traffic Poisson flows scheduled!"
        
        # Verify dummy environment step test
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
        assert len(obs) > 0
        
        # Step twice and verify observations are correct
        actions = {agent_id: 0 for agent_id in env.agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(next_obs) == len(obs)
        env.close()
        
    finally:
        # Clean up files
        try:
            if os.path.exists(net_file):
                os.remove(net_file)
        except Exception:
            pass
            
        try:
            if os.path.exists(rou_file):
                os.remove(rou_file)
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
