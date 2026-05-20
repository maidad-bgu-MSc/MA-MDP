import os
import xml.etree.ElementTree as ET
import pytest
from simulator.problem_generator import SCENARIOS, generate_problem


def test_scenario_names():
    assert set(SCENARIOS) == {"baseline", "dense_wave", "cross_surge", "split_rush"}


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        generate_problem("nonexistent_scenario")


def test_generate_all_scenarios(tmp_path):
    """Verifies each scenario writes a parseable XML route file with at least one flow.
    Uses tmp_path so tests don't pollute the working directory.
    Net file generation is skipped if wave_1x4.net.xml already exists in CWD
    (avoids requiring SUMO during CI).
    """
    # Patch _ensure_net_file to return the existing net file without re-generating
    import simulator.problem_generator as pg
    original_ensure = pg._ensure_net_file

    def mock_ensure(out_dir):
        net = "wave_1x4.net.xml"
        if not os.path.exists(net):
            pytest.skip("wave_1x4.net.xml not found — SUMO required to generate it")
        return net

    pg._ensure_net_file = mock_ensure
    try:
        for scenario in SCENARIOS:
            _, rou_file = generate_problem(scenario, out_dir=str(tmp_path))
            assert os.path.exists(rou_file), f"Route file missing for {scenario}: {rou_file}"
            tree = ET.parse(rou_file)
            root = tree.getroot()
            flows = root.findall("flow")
            assert len(flows) > 0, f"No <flow> elements in {rou_file}"
    finally:
        pg._ensure_net_file = original_ensure


def test_split_rush_flow_order(tmp_path):
    """Verifies split_rush route file has flows sorted by begin time (SUMO requirement)."""
    import simulator.problem_generator as pg
    original_ensure = pg._ensure_net_file

    def mock_ensure(out_dir):
        net = "wave_1x4.net.xml"
        if not os.path.exists(net):
            pytest.skip("wave_1x4.net.xml not found")
        return net

    pg._ensure_net_file = mock_ensure
    try:
        _, rou_file = generate_problem("split_rush", out_dir=str(tmp_path))
        tree = ET.parse(rou_file)
        root = tree.getroot()
        begin_times = [float(f.get("begin", 0)) for f in root.findall("flow")]
        assert begin_times == sorted(begin_times), "split_rush flows are not sorted by begin time"
    finally:
        pg._ensure_net_file = original_ensure