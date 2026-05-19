import os
import sys

# Setup SUMO paths programmatically at the very beginning of the test suite
def setup_sumo_env():
    try:
        import sumo
        sumo_home = os.path.abspath(os.path.dirname(sumo.__file__))
        os.environ["SUMO_HOME"] = sumo_home
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME")

    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")
    
    paths = []
    if sumo_home:
        paths.append(os.path.join(sumo_home, "bin"))
    paths.append(scripts_dir)
    
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + current_path

setup_sumo_env()

# Now it is safe to import sumo_rl and other dependencies
import shutil
import xml.etree.ElementTree as ET
import pytest
import sumo_rl

from simulator.generate_1x4_wave import build_1x4_scenario
from simulator.env_setup import make_wave_env, GlobalRewardWrapper, QueueObservationFunction, global_reward_fn

def get_traffic_signals(env):
    """Recursively unwraps any wrapped environment to retrieve the core traffic_signals dictionary."""
    curr = env
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        if hasattr(curr, "traffic_signals"):
            return curr.traffic_signals
        
        # Try various standard unwrapping properties in order
        next_curr = None
        for attr in ["env", "aec_env", "raw_env"]:
            val = getattr(curr, attr, None)
            if val is not None and id(val) not in visited:
                next_curr = val
                break
        curr = next_curr
    raise ValueError("Could not find traffic_signals in environment wrappers!")

def test_1x4_network_generation():
    """Asserts that 1x4 arterial corridor network is compiled and has the correct topology using netgenerate."""
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the 1x4 scenario
        net, rou = build_1x4_scenario()
        
        assert net == net_file
        assert rou == rou_file
        assert os.path.exists(net_file)
        assert os.path.exists(rou_file)
        
        # Verify node/edge properties via XML parsing of net file
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # We expect exactly 4 traffic lights (A0, B0, C0, D0)
        tl_logics = root.findall("tlLogic")
        tl_ids = {tl.get("id") for tl in tl_logics}
        assert tl_ids == {"A0", "B0", "C0", "D0"}, f"Expected traffic light IDs A0, B0, C0, D0; got {tl_ids}"
        
        # Verify arterial edge A0B0 exists
        edges = root.findall("edge")
        a0b0_edges = [e for e in edges if e.get("id") == "A0B0"]
        assert len(a0b0_edges) == 1, "Arterial edge A0B0 not found!"
        
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

def test_aec_dummy_agent_loop():
    """Test A: Runs a full 3600-step dummy agent episode using standard PettingZoo AEC loop without exceptions."""
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the 1x4 scenario
        build_1x4_scenario()
        
        env = make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=3600, delta_time=5, use_gui=False)
        env.reset()
        
        # Robustly retrieve the core traffic_signals dictionary and mock halting count to 0 to prevent gridlock resets
        traffic_signals = get_traffic_signals(env)
        for ts_id, ts in traffic_signals.items():
            ts.sumo.lane.getLastStepHaltingNumber = lambda lane_id: 0
            
        steps = 0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()
                
            env.step(action)
            steps += 1
            
        env.close()
        assert steps > 0, "AEC environment did not run any steps!"
        
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

def test_global_reward_synchronization():
    """Test B: Steps the environment forward by 100 ticks (20 steps * 5s) and asserts global reward synchronization."""
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the 1x4 scenario
        build_1x4_scenario()
        
        parallel_env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=rou_file,
            use_gui=False,
            num_seconds=500,
            delta_time=5,
            reward_fn=global_reward_fn,
            observation_class=QueueObservationFunction
        )
        wrapped = GlobalRewardWrapper(parallel_env)
        obs, infos = wrapped.reset()
        
        assert set(wrapped.agents) == {"A0", "B0", "C0", "D0"}
        
        # Step forward 20 times (20 steps * 5s = 100 ticks/seconds)
        for step in range(20):
            actions = {agent_id: wrapped.action_space(agent_id).sample() for agent_id in wrapped.agents}
            obs, rewards, terminations, truncations, infos = wrapped.step(actions)
            
            # Assert all agents receive identical global reward on every step
            first_reward = list(rewards.values())[0]
            for agent_id, reward in rewards.items():
                assert reward == first_reward, f"Reward mismatch: {agent_id} got {reward}, expected {first_reward}"
                
        wrapped.close()
        
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

def test_early_termination_trigger():
    """Test C: Forces queue length to exceed 30, asserts terminal penalty (-10000) and simultaneous early termination flags."""
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Build the 1x4 scenario
        build_1x4_scenario()
        
        parallel_env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=rou_file,
            use_gui=False,
            num_seconds=200,
            delta_time=5,
            reward_fn=global_reward_fn,
            observation_class=QueueObservationFunction
        )
        wrapped = GlobalRewardWrapper(parallel_env)
        obs, infos = wrapped.reset()
        
        # Robustly mock TraCI halting numbers to simulate a gridlock (35 vehicles > 30)
        traffic_signals = get_traffic_signals(wrapped)
        for ts_id, ts in traffic_signals.items():
            ts.sumo.lane.getLastStepHaltingNumber = lambda lane_id: 35
            
        actions = {agent_id: wrapped.action_space(agent_id).sample() for agent_id in wrapped.agents}
        next_obs, rewards, terminations, truncations, infos = wrapped.step(actions)
        
        # Assert massive terminal penalty and early termination flags are simultaneously active
        for agent_id in wrapped.agents:
            assert terminations[agent_id] is True, f"Agent {agent_id} did not terminate early on gridlock!"
            assert rewards[agent_id] == -10000.0, f"Agent {agent_id} did not receive terminal penalty!"
            
        wrapped.close()
        
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
