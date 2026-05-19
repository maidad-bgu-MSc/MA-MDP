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

def discretize_queue_5_bins(q):
    """5-Bin Discretization for queue lengths:
    0: 0 cars
    1: 1-5 cars
    2: 6-15 cars
    3: 16-29 cars
    4: 30+ cars
    """
    if q <= 0:
        return 0
    elif q <= 5:
        return 1
    elif q <= 15:
        return 2
    elif q <= 29:
        return 3
    else:
        return 4

class QueueObservationFunction(ObservationFunction):
    """Custom look-ahead observation function perfectly sized for Tabular Q-Learning.
    
    Observes exactly 2 values for each agent (5x5 = 25 state space):
    1. Local queue length on the East-West arterial incoming lane (discretized to 5 bins).
    2. Queue length on the immediate upstream East-West arterial incoming lane (neighbor to the West, discretized to 5 bins).
    """
    
    def __init__(self, ts):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        ts_id = self.ts.id
        
        # Determine local and upstream East-West arterial incoming lanes based on traffic light ID
        # Flow runs West to East (Eastbound)
        if ts_id == "A0":
            local_lane = "left0A0_0"
            upstream_lane = None
        elif ts_id == "B0":
            local_lane = "A0B0_0"
            upstream_lane = "left0A0_0"
        elif ts_id == "C0":
            local_lane = "B0C0_0"
            upstream_lane = "A0B0_0"
        elif ts_id == "D0":
            local_lane = "C0D0_0"
            upstream_lane = "B0C0_0"
        else:
            raise ValueError(f"Unknown traffic light ID: {ts_id}")
            
        # Get queue lengths (halting vehicles count)
        local_q = self.ts.sumo.lane.getLastStepHaltingNumber(local_lane)
        upstream_q = 0
        if upstream_lane is not None:
            upstream_q = self.ts.sumo.lane.getLastStepHaltingNumber(upstream_lane)
            
        # Apply 5-bin discretization
        discrete_local = discretize_queue_5_bins(local_q)
        discrete_upstream = discretize_queue_5_bins(upstream_q)
        
        return np.array([discrete_local, discrete_upstream], dtype=np.int32)
        
    def observation_space(self) -> gym.spaces.Box:
        # Array of 2 discrete bins (values from 0 to 4)
        return gym.spaces.Box(
            low=np.zeros(2, dtype=np.int32),
            high=np.full(2, 4, dtype=np.int32),
            dtype=np.int32
        )

def global_reward_fn(ts):
    """Calculates individual negative waiting times at the intersection."""
    return -float(sum(ts.get_accumulated_waiting_time_per_lane()))

class GlobalRewardWrapper:
    """Wrapper that enforces global reward synchronization and safety gridlock resets.
    
    1. Sums all individual negative waiting times and broadcasts this single scalar to all agents.
    2. Gridlock Reset: If any single queue hits Bin 4 (30+ cars), sets terminations to True for all agents
       and applies a massive terminal penalty of -10,000.
    """
    def __init__(self, env):
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents
        
    @property
    def agents(self):
        return self.env.agents
        
    @property
    def observation_spaces(self):
        return self.env.observation_spaces
        
    @property
    def action_spaces(self):
        return self.env.action_spaces
        
    def observation_space(self, agent):
        return self.env.observation_space(agent)
        
    def action_space(self, agent):
        return self.env.action_space(agent)
        
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
        
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Gridlock Reset check: if any queue hits Bin 4 (30+ cars, meaning obs[0] >= 4)
        gridlock = False
        for agent_id, agent_obs in obs.items():
            if agent_obs[0] >= 4:
                gridlock = True
                break
                
        if gridlock:
            # Enforce immediate early termination across all agents
            terminations = {agent: True for agent in rewards.keys()}
            # Apply a massive terminal penalty
            synchronized_rewards = {agent: -10000.0 for agent in rewards.keys()}
        else:
            # Broadcast the sum of negative waiting times as the global reward
            global_reward = sum(rewards.values())
            synchronized_rewards = {agent: global_reward for agent in rewards.keys()}
            
        return obs, synchronized_rewards, terminations, truncations, infos
        
    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()

def make_wave_env(net_file="wave_1x4.net.xml", route_file="wave_1x4.rou.xml", num_seconds=3600, delta_time=5, out_csv_name=None, use_gui=False):
    """Initializes and wraps the sumo-rl environment into a PettingZoo AEC interface with look-ahead obs, global reward, and gridlock reset."""
    # Create the parallel environment with min_green = 10 safety rail
    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=delta_time,
        min_green=10,
        reward_fn=global_reward_fn,
        observation_class=QueueObservationFunction,
        out_csv_name=out_csv_name
    )
    
    # Wrap it to enforce global reward broadcast & early gridlock reset
    wrapped_parallel = GlobalRewardWrapper(parallel_env)
    
    # Convert to AEC turn-based interface for compatibility
    aec_env = parallel_to_aec(wrapped_parallel)
    return aec_env

if __name__ == "__main__":
    print("Testing environment initialization...")
    env = make_wave_env(num_seconds=500)
    env.reset()
    print("Environment successfully initialized!")
    print(f"Agents in environment: {env.agents}")
    
    first_agent = env.agents[0]
    print(f"Agent '{first_agent}' observation space: {env.observation_space(first_agent)}")
    print(f"Agent '{first_agent}' action space: {env.action_space(first_agent)}")
    env.close()
