import os
import sys
import numpy as np
import gymnasium as gym

# Setup SUMO paths programmatically
_SUMO_ENV_INITIALIZED = False

def setup_sumo_env():
    global _SUMO_ENV_INITIALIZED
    if _SUMO_ENV_INITIALIZED:
        return

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
    existing = current_path.split(os.pathsep)
    to_prepend = [p for p in paths if p not in existing]
    if to_prepend:
        os.environ["PATH"] = os.pathsep.join(to_prepend) + os.pathsep + current_path

    _SUMO_ENV_INITIALIZED = True

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

def is_vertical_lane(ts, lane_id):
    """Classifies a lane as vertical (North-South) or horizontal (East-West) based on its geometry.
    Falls back to name heuristics if geometry cannot be fetched.
    """
    try:
        shape = ts.sumo.lane.getShape(lane_id)
        if isinstance(shape, str):
            # Parse string shape: "x1,y1,z1 x2,y2,z2" or "x1,y1 x2,y2"
            points = [p.split(',') for p in shape.split()]
            coords = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
        else:
            coords = [(p[0], p[1]) for p in shape]
        if len(coords) >= 2:
            x1, y1 = coords[0]
            x2, y2 = coords[-1]
            return abs(y2 - y1) > abs(x2 - x1)
    except Exception as e:
        pass
    
    # Fallbacks based on naming patterns
    edge_id = lane_id.split("_")[0]
    if "top" in edge_id or "bottom" in edge_id:
        return True
    if "left" in edge_id or "right" in edge_id:
        return False
    if len(edge_id) >= 4:
        return edge_id[0] == edge_id[2]
    return False

class QueueObservationFunctionResco(ObservationFunction):
    """Look-ahead observation function for RESCO scenarios.
    
    Observes 4 values for each agent (5^4 = 625 state space):
    1. Local East-West queue (discretized to 5 bins).
    2. Local North-South queue (discretized to 5 bins).
    3. Rest of Network East-West queue (discretized to 5 bins).
    4. Rest of Network North-South queue (discretized to 5 bins).
    """
    
    def __init__(self, ts):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        # Sum up all local incoming lane queues
        local_lanes = self.ts.lanes
        local_ew, local_ns = 0, 0
        
        for lane in local_lanes:
            q = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
            if is_vertical_lane(self.ts, lane):
                local_ns += q
            else:
                local_ew += q
                
        # Sum up all rest-of-network queues, then AVERAGE per other-junction so the
        # magnitude stays comparable to a single local junction. Raw summing over many
        # neighbours (grid4x4 has 15) saturates the top bin (30+) almost every step, which
        # makes the two "rest-of-network" state dimensions constant and uninformative —
        # collapsing the 625-state table to local-only and preventing any coordination.
        other_ew, other_ns = 0.0, 0.0
        num_others = 0
        for other_ts_id in self.ts.env.ts_ids:
            if other_ts_id == self.ts.id:
                continue
            num_others += 1
            other_ts = self.ts.env.traffic_signals[other_ts_id]
            for lane in other_ts.lanes:
                q = self.ts.sumo.lane.getLastStepHaltingNumber(lane)
                if is_vertical_lane(self.ts, lane):
                    other_ns += q
                else:
                    other_ew += q

        if num_others > 0:
            other_ew /= num_others
            other_ns /= num_others

        # Apply 5-bin discretization to all 4 dimensions
        obs = np.array([
            discretize_queue_5_bins(local_ew),
            discretize_queue_5_bins(local_ns),
            discretize_queue_5_bins(other_ew),
            discretize_queue_5_bins(other_ns)
        ], dtype=np.int32)
        
        return obs
        
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.zeros(4, dtype=np.int32),
            high=np.full(4, 4, dtype=np.int32),
            dtype=np.int32
        )

class MaxPressureObservationFunctionResco(ObservationFunction):
    """Phase-aware observation for multi-phase RESCO junctions (cologne3: 3-4 phases,
    grid4x4: 8). Reports 4 discrete values per agent:

        0. most-demanded green phase  -- argmax over phases of the halting queue on the
           lanes that phase gives green to. This is the key signal the old EW/NS encoding
           lacked: it tells the agent WHICH of its N phases relieves the current demand,
           so the state maps directly onto the phase-index action (cf. max-pressure control).
        1. current green phase index  -- lets the agent account for min_green / switching.
        2. local congestion bin       -- total local halting queue, 5-bin discretized.
        3. rest-of-network bin        -- avg per-junction halting queue, 5-bin discretized.

    Encoded to a single state index by marl_algorithms.get_resco_state (radix [8,8,5,5]).
    """

    def __init__(self, ts):
        super().__init__(ts)
        # ts.lanes / ts.green_phases don't exist yet at construction (TrafficSignal builds
        # them right AFTER instantiating the observation_fn), so build the phase->lane map
        # lazily on first call.
        self._phase_lanes = None

    def _build_phase_lanes(self):
        """Map each green-phase index -> set of incoming lanes it gives green ('g'/'G') to.
        The phase state-string index aligns with getControlledLinks() index."""
        ts = self.ts
        links = ts.sumo.trafficlight.getControlledLinks(ts.id)
        phase_lanes = []
        for phase in ts.green_phases:
            served = set()
            for i, sig in enumerate(phase.state):
                if sig in ("g", "G") and i < len(links) and links[i]:
                    served.add(links[i][0][0])  # incoming lane of the first connection
            phase_lanes.append(served)
        return phase_lanes

    def __call__(self) -> np.ndarray:
        ts = self.ts
        # Under fixed_ts (the SUMO-native baseline) TrafficSignal._build_phases returns
        # early and never sets green_phases; that baseline ignores observations anyway,
        # so return a neutral vector instead of crashing.
        if not hasattr(ts, "green_phases"):
            return np.zeros(4, dtype=np.int32)
        if self._phase_lanes is None:
            self._phase_lanes = self._build_phase_lanes()

        local_q = {lane: ts.sumo.lane.getLastStepHaltingNumber(lane) for lane in ts.lanes}
        phase_queues = [sum(local_q.get(l, 0) for l in served) for served in self._phase_lanes]
        local_max_phase = int(np.argmax(phase_queues)) if phase_queues else 0
        current_phase = int(getattr(ts, "green_phase", 0))
        total_local = sum(local_q.values())

        # Average per-junction rest-of-network queue (same rationale as the EW/NS obs:
        # averaging keeps the magnitude comparable to a single junction so the bin is
        # informative rather than pinned at the top for large nets).
        other_total, num_others = 0.0, 0
        for other_id in ts.env.ts_ids:
            if other_id == ts.id:
                continue
            num_others += 1
            other_ts = ts.env.traffic_signals[other_id]
            other_total += sum(ts.sumo.lane.getLastStepHaltingNumber(l) for l in other_ts.lanes)
        if num_others > 0:
            other_total /= num_others

        return np.array([
            local_max_phase,
            current_phase,
            discretize_queue_5_bins(total_local),
            discretize_queue_5_bins(other_total),
        ], dtype=np.int32)

    def observation_space(self) -> gym.spaces.Box:
        # Phase dims bounded by RESCO_MAX_PHASES-1 (=7; grid4x4 has the most, 8 phases);
        # congestion dims by the 5-bin discretizer (0..4).
        return gym.spaces.Box(
            low=np.zeros(4, dtype=np.int32),
            high=np.array([7, 7, 4, 4], dtype=np.int32),
            dtype=np.int32,
        )

def global_reward_fn(ts):
    """Calculates individual negative waiting times at the intersection."""
    return -float(sum(ts.get_accumulated_waiting_time_per_lane()))

class GlobalRewardWrapperResco:
    """Wrapper that enforces global reward synchronization for RESCO environments."""
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
        global_reward = sum(rewards.values())
        synchronized_rewards = {agent: global_reward for agent in rewards.keys()}
        return obs, synchronized_rewards, terminations, truncations, infos
        
    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()

def make_resco_env(scenario_name, num_seconds=3600, delta_time=5, out_csv_name=None,
                   use_gui=False, sumo_seed="random", fixed_ts=False):
    """Initializes and wraps a RESCO environment into a PettingZoo AEC interface.

    sumo_seed: int for reproducible SUMO stochasticity (vehicle departures, etc.), or "random".
    fixed_ts: if True, signals follow the net file's built-in fixed-time program and ignore actions
              (used for the SUMO-native fixed-time baseline).
    """
    net_file = os.path.join("simulator", "resco_environments", scenario_name, f"{scenario_name}.net.xml")
    route_file = os.path.join("simulator", "resco_environments", scenario_name, f"{scenario_name}.rou.xml")

    if not os.path.exists(net_file) or not os.path.exists(route_file):
        # Auto-download if missing
        from simulator.download_resco_scenarios import download_resco_data
        download_resco_data()

    begin_time = 23500 if scenario_name == "cologne3" else 0

    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        begin_time=begin_time,
        num_seconds=num_seconds,
        delta_time=delta_time,
        min_green=10,
        reward_fn=global_reward_fn,
        observation_class=MaxPressureObservationFunctionResco,
        out_csv_name=out_csv_name,
        sumo_seed=sumo_seed,
        fixed_ts=fixed_ts,
    )

    wrapped_parallel = GlobalRewardWrapperResco(parallel_env)
    aec_env = parallel_to_aec(wrapped_parallel)
    return aec_env

def make_resco_parallel_env(scenario_name, num_seconds=3600, delta_time=5, use_gui=False,
                            sumo_seed="random", fixed_ts=False):
    """Initializes and returns the wrapped parallel sumo-rl environment (for VDN training).

    sumo_seed / fixed_ts: see make_resco_env.
    """
    net_file = os.path.join("simulator", "resco_environments", scenario_name, f"{scenario_name}.net.xml")
    route_file = os.path.join("simulator", "resco_environments", scenario_name, f"{scenario_name}.rou.xml")

    if not os.path.exists(net_file) or not os.path.exists(route_file):
        from simulator.download_resco_scenarios import download_resco_data
        download_resco_data()

    begin_time = 23500 if scenario_name == "cologne3" else 0

    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        begin_time=begin_time,
        num_seconds=num_seconds,
        delta_time=delta_time,
        min_green=10,
        reward_fn=global_reward_fn,
        observation_class=MaxPressureObservationFunctionResco,
        sumo_seed=sumo_seed,
        fixed_ts=fixed_ts,
    )
    return GlobalRewardWrapperResco(parallel_env)

def make_cologne3_env(use_gui=False, out_csv_name=None, num_seconds=3600):
    return make_resco_env("cologne3", num_seconds=num_seconds, out_csv_name=out_csv_name, use_gui=use_gui)

def make_grid4x4_env(use_gui=False, out_csv_name=None, num_seconds=3600):
    return make_resco_env("grid4x4", num_seconds=num_seconds, out_csv_name=out_csv_name, use_gui=use_gui)

if __name__ == "__main__":
    print("Testing Cologne3 initialization...")
    env = make_cologne3_env(num_seconds=200)
    env.reset()
    print(f"Cologne3 success! Agents: {env.agents}")
    env.close()
    
    print("Testing Grid4x4 initialization...")
    env = make_grid4x4_env(num_seconds=200)
    env.reset()
    print(f"Grid4x4 success! Agents: {env.agents}")
    env.close()
