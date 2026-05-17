import os
import sys
import numpy as np
import gymnasium as gym

# Setup SUMO paths programmatically
def setup_sumo_env():
    """Locates sumo / netgenerate executables and sets SUMO_HOME programmatically."""
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

# Now import sumo-rl, pettingzoo, and other required packages
from sumo_rl.environment.observations import ObservationFunction
import sumo_rl
from pettingzoo.utils.conversions import parallel_to_aec

class QueueObservationFunction(ObservationFunction):
    """Custom observation function that strictly returns incoming queue lengths."""
    
    def __init__(self, ts):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        # self.ts.get_lanes_queue() returns the halting vehicle counts on all incoming lanes
        queue = self.ts.get_lanes_queue()
        return np.array(queue, dtype=np.float32)
        
    def observation_space(self) -> gym.spaces.Box:
        num_lanes = len(self.ts.lanes)
        return gym.spaces.Box(
            low=np.zeros(num_lanes, dtype=np.float32),
            high=np.full(num_lanes, np.inf, dtype=np.float32),
            dtype=np.float32
        )

def custom_reward_fn(ts):
    """Strict MDP Reward: Negative accumulated waiting time of all vehicles at the specific intersection."""
    # Sum the accumulated waiting time on all lanes managed by the traffic signal
    return -float(sum(ts.get_accumulated_waiting_time_per_lane()))

def make_env(net_file="grid.net.xml", route_file="grid.rou.xml", num_seconds=3600, delta_time=5, out_csv_name=None, use_gui=False):
    """Initializes and wraps the sumo-rl environment into a PettingZoo AEC interface."""
    # Create the parallel environment first
    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        reward_fn=custom_reward_fn,
        observation_class=QueueObservationFunction,
        out_csv_name=out_csv_name
    )
    # Convert it to AEC turn-based API for full Tianshou compatibility
    aec_env = parallel_to_aec(parallel_env)
    return aec_env

if __name__ == "__main__":
    print("Testing environment initialization...")
    env = make_env(num_seconds=500)
    env.reset()
    print("Environment successfully initialized!")
    print(f"Agents in environment: {env.agents}")
    
    # Check the observation and action spaces of the first agent
    first_agent = env.agents[0]
    print(f"Agent '{first_agent}' observation space: {env.observation_space(first_agent)}")
    print(f"Agent '{first_agent}' action space: {env.action_space(first_agent)}")
    env.close()
