import os
import sys

# Setup SUMO paths programmatically
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

import pytest
from simulator.env_setup_resco import make_cologne3_env, make_grid4x4_env

def test_resco_downloads():
    """Verifies that RESCO scenario files are successfully downloaded and placed in the target paths."""
    scenarios = ["cologne3", "grid4x4"]
    for scenario in scenarios:
        net_path = os.path.join("simulator", "resco_environments", scenario, f"{scenario}.net.xml")
        rou_path = os.path.join("simulator", "resco_environments", scenario, f"{scenario}.rou.xml")
        assert os.path.exists(net_path), f"Missing net file: {net_path}"
        assert os.path.exists(rou_path), f"Missing route file: {rou_path}"

def run_aec_dummy_loop(env):
    """Runs a short loop of steps using a dummy policy in the provided AEC environment."""
    env.reset()
    steps = 0
    # Run 40 ticks/agent steps to verify stepping
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        steps += 1
        if steps >= 40:
            break
    env.close()
    assert steps >= 40

def test_resco_cologne3_aec_loop():
    """Initializes cologne3 and verifies stepping dynamics in a dummy AEC loop."""
    env = make_cologne3_env(num_seconds=500)
    run_aec_dummy_loop(env)

def test_resco_grid4x4_aec_loop():
    """Initializes grid4x4 and verifies stepping dynamics in a dummy AEC loop."""
    env = make_grid4x4_env(num_seconds=500)
    run_aec_dummy_loop(env)
