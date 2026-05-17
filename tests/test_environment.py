import os
import shutil
import pytest

from scale_network import setup_sumo_env
setup_sumo_env()

import sumo_rl
from scale_network import run_netgenerate, modify_traffic_lights, generate_routes
from env_setup import QueueObservationFunction, custom_reward_fn

def test_environment_stepping():
    """Validates PettingZoo parallel environment stepping, agent naming, and observation structure."""
    net_file = "grid_2x2.net.xml"
    rou_file = "grid_2x2.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    # Back up existing files if they exist to keep workspace pristine
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build temp files for the environment to run
        setup_sumo_env()
        run_netgenerate(size=2)
        modify_traffic_lights(net_file)
        generate_routes(size=2, rou_file=rou_file)
        
        # Initialize parallel env
        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=rou_file,
            use_gui=False,
            num_seconds=100,  # Short duration
            delta_time=5,
            reward_fn=custom_reward_fn,
            observation_class=QueueObservationFunction
        )
        
        # Assert environment initialization
        obs_dict, infos = env.reset()
        assert obs_dict is not None, "env.reset() returned None"
        assert len(env.possible_agents) == 4, f"Expected 4 agents on a 2x2 grid, found {len(env.possible_agents)}"
        
        # Validate observation and action space schemas
        for agent_id in env.possible_agents:
            obs = obs_dict[agent_id]
            # Custom queue observations return length 2 vector (for simple 2x2 junctions)
            assert len(obs) == 2, f"Expected observation size of 2, got {len(obs)}"
            
            action_space = env.action_space(agent_id)
            assert hasattr(action_space, "n"), f"Action space for {agent_id} is not discrete!"
        
        # Step the environment 5 times with random valid actions
        for _ in range(5):
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = int(env.action_space(agent_id).sample())
                
            next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
            
            assert next_obs is not None
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(next_infos, dict)
            
        env.close()
        
    finally:
        # Clean up temporary test files
        if os.path.exists(net_file):
            os.remove(net_file)
        if os.path.exists(rou_file):
            os.remove(rou_file)
            
        # Restore original backups
        if os.path.exists(net_backup):
            shutil.move(net_backup, net_file)
        if os.path.exists(rou_backup):
            shutil.move(rou_backup, rou_file)
