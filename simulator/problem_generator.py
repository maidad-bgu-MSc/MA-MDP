import os
import sys

SCENARIOS = ["baseline", "dense_wave", "cross_surge", "split_rush"]
NET_FILE = "wave_1x4.net.xml"

# Shared route topology (edges) — identical across all scenarios
_ROUTES = {
    "r_eastbound": ["left0A0", "A0B0", "B0C0", "C0D0", "D0right0"],
    "r_westbound": ["right0D0", "D0C0", "C0B0", "B0A0", "A0left0"],
    "r_ns_0": ["top0A0", "A0bottom0"],
    "r_sn_0": ["bottom0A0", "A0top0"],
    "r_ns_1": ["top1B0", "B0bottom1"],
    "r_sn_1": ["bottom1B0", "B0top1"],
    "r_ns_2": ["top2C0", "C0bottom2"],
    "r_sn_2": ["bottom2C0", "C0top2"],
    "r_ns_3": ["top3D0", "D0bottom3"],
    "r_sn_3": ["bottom3D0", "D0top3"],
}

_CROSS_ROUTES = ["r_ns_0", "r_sn_0", "r_ns_1", "r_sn_1", "r_ns_2", "r_sn_2", "r_ns_3", "r_sn_3"]


def _write_header(f):
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
    f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
            'length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
    for r_id, edges in _ROUTES.items():
        f.write(f'    <route id="{r_id}" edges="{" ".join(edges)}"/>\n')
    f.write("\n")


def _write_sorted_flows(f, flows):
    """Write (begin, xml_string) tuples sorted by begin time — SUMO requires sorted departures."""
    for _, line in sorted(flows, key=lambda x: x[0]):
        f.write(line)


def _ensure_net_file(out_dir):
    net_path = os.path.join(out_dir, NET_FILE) if out_dir != "." else NET_FILE
    if not os.path.exists(net_path):
        from simulator.generate_1x4_wave import setup_sumo_env, run_netgenerate, modify_traffic_lights
        setup_sumo_env()
        run_netgenerate(net_path)
        modify_traffic_lights(net_path)
    return net_path


def _generate_baseline_routes(rou_file):
    from simulator.generate_1x4_wave import generate_wave_routes
    generate_wave_routes(rou_file)


def _generate_dense_wave_routes(rou_file):
    """50-vehicle platoons every 100s; NS prob=0.05. Forces sequential EW cooperation."""
    flows = []
    # NS cross-traffic (begin=0, written first so begin-sort keeps them first)
    for idx, r_id in enumerate(_CROSS_ROUTES):
        flows.append((0, f'    <flow id="cross_flow_{idx}" type="car" begin="0" end="3600" '
                         f'probability="0.05" route="{r_id}" departLane="best" departSpeed="max"/>\n'))
    # EW platoons every 100s
    for t in range(0, 3600, 100):
        flows.append((t, f'    <flow id="platoon_eb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="50" route="r_eastbound" departLane="best" departSpeed="max"/>\n'))
        flows.append((t, f'    <flow id="platoon_wb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="50" route="r_westbound" departLane="best" departSpeed="max"/>\n'))

    with open(rou_file, "w") as f:
        _write_header(f)
        _write_sorted_flows(f, flows)
        f.write('</routes>\n')
    print(f"Generated dense_wave routes: {rou_file}")


def _generate_cross_surge_routes(rou_file):
    """20-vehicle platoons every 180s; NS prob=0.4. Forces collective EW sacrifice."""
    flows = []
    for idx, r_id in enumerate(_CROSS_ROUTES):
        flows.append((0, f'    <flow id="cross_flow_{idx}" type="car" begin="0" end="3600" '
                         f'probability="0.4" route="{r_id}" departLane="best" departSpeed="max"/>\n'))
    for t in range(0, 3600, 180):
        flows.append((t, f'    <flow id="platoon_eb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="20" route="r_eastbound" departLane="best" departSpeed="max"/>\n'))
        flows.append((t, f'    <flow id="platoon_wb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="20" route="r_westbound" departLane="best" departSpeed="max"/>\n'))

    with open(rou_file, "w") as f:
        _write_header(f)
        _write_sorted_flows(f, flows)
        f.write('</routes>\n')
    print(f"Generated cross_surge routes: {rou_file}")


def _generate_split_rush_routes(rou_file):
    """Phase 1 (0-1800s): 40-vehicle EW platoons every 120s + NS prob=0.05.
    Phase 2 (1800-3600s): no EW platoons + NS prob=0.5.
    Forces collective strategy switch mid-episode."""
    flows = []
    # Phase 1 NS (light)
    for idx, r_id in enumerate(_CROSS_ROUTES):
        flows.append((0, f'    <flow id="cross_p1_{idx}" type="car" begin="0" end="1800" '
                         f'probability="0.05" route="{r_id}" departLane="best" departSpeed="max"/>\n'))
    # Phase 1 EW platoons
    for t in range(0, 1800, 120):
        flows.append((t, f'    <flow id="platoon_eb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="40" route="r_eastbound" departLane="best" departSpeed="max"/>\n'))
        flows.append((t, f'    <flow id="platoon_wb_{t}" type="car" begin="{t}" end="{t+10}" '
                         f'number="40" route="r_westbound" departLane="best" departSpeed="max"/>\n'))
    # Phase 2 NS (heavy) — no EW platoons
    for idx, r_id in enumerate(_CROSS_ROUTES):
        flows.append((1800, f'    <flow id="cross_p2_{idx}" type="car" begin="1800" end="3600" '
                            f'probability="0.5" route="{r_id}" departLane="best" departSpeed="max"/>\n'))

    with open(rou_file, "w") as f:
        _write_header(f)
        _write_sorted_flows(f, flows)
        f.write('</routes>\n')
    print(f"Generated split_rush routes: {rou_file}")


_GENERATORS = {
    "baseline": _generate_baseline_routes,
    "dense_wave": _generate_dense_wave_routes,
    "cross_surge": _generate_cross_surge_routes,
    "split_rush": _generate_split_rush_routes,
}

_ROU_NAMES = {
    "baseline": "wave_1x4_baseline.rou.xml",
    "dense_wave": "wave_1x4_dense_wave.rou.xml",
    "cross_surge": "wave_1x4_cross_surge.rou.xml",
    "split_rush": "wave_1x4_split_rush.rou.xml",
}


def generate_problem(scenario_name: str, out_dir: str = ".") -> tuple:
    """Generates (or reuses) net+route files for the named scenario.
    Returns (net_file_path, rou_file_path).
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Choose from {SCENARIOS}")

    net_file = _ensure_net_file(out_dir)
    rou_file = os.path.join(out_dir, _ROU_NAMES[scenario_name]) if out_dir != "." else _ROU_NAMES[scenario_name]

    if not os.path.exists(rou_file):
        _GENERATORS[scenario_name](rou_file)

    return net_file, rou_file


def make_problem_env(scenario_name: str, num_seconds: int = 3600, out_csv_name=None):
    """Returns an AEC PettingZoo env for the given scenario (for tabular training/eval)."""
    from simulator.env_setup import make_wave_env
    net_file, rou_file = generate_problem(scenario_name)
    return make_wave_env(net_file=net_file, route_file=rou_file,
                         num_seconds=num_seconds, out_csv_name=out_csv_name)


def make_problem_parallel_env(scenario_name: str, num_seconds: int = 3600):
    """Returns GlobalRewardWrapper(parallel_env) for VDN/QMIX training."""
    import sumo_rl
    from simulator.env_setup import GlobalRewardWrapper, QueueObservationFunction, global_reward_fn, setup_sumo_env
    setup_sumo_env()
    net_file, rou_file = generate_problem(scenario_name)
    parallel_env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=rou_file,
        use_gui=False,
        num_seconds=num_seconds,
        delta_time=5,
        min_green=10,
        reward_fn=global_reward_fn,
        observation_class=QueueObservationFunction,
    )
    return GlobalRewardWrapper(parallel_env)


if __name__ == "__main__":
    for scenario in SCENARIOS:
        net, rou = generate_problem(scenario)
        print(f"  {scenario}: net={net}, rou={rou}")
    print("All scenarios generated successfully.")