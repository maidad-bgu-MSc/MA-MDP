import pytest
import gymnasium as gym

def clamp_action_safe(action_val, action_space):
    """Encapsulates the safe action modulo wrapping pipeline used in the orchestration loops."""
    # Modulo mapping clamps values gracefully within bounds
    return int(action_val % action_space.n)

def test_corner_junction_discrete_one_clamping():
    """Asserts that corner junctions with Discrete(1) spaces always clamp to action 0."""
    action_space_corner = gym.spaces.Discrete(1)
    
    # Passing any action (including out-of-bounds) should map to 0
    assert clamp_action_safe(action_val=0, action_space=action_space_corner) == 0
    assert clamp_action_safe(action_val=1, action_space=action_space_corner) == 0
    assert clamp_action_safe(action_val=5, action_space=action_space_corner) == 0
    assert clamp_action_safe(action_val=-1, action_space=action_space_corner) == 0

def test_boundary_junction_discrete_two_clamping():
    """Asserts that boundary/center junctions with Discrete(2) spaces clamp via modulo."""
    action_space_standard = gym.spaces.Discrete(2)
    
    # Valid actions
    assert clamp_action_safe(action_val=0, action_space=action_space_standard) == 0
    assert clamp_action_safe(action_val=1, action_space=action_space_standard) == 1
    
    # Out of bounds actions -> map to 0 or 1 safely
    assert clamp_action_safe(action_val=2, action_space=action_space_standard) == 0
    assert clamp_action_safe(action_val=3, action_space=action_space_standard) == 1
    assert clamp_action_safe(action_val=10, action_space=action_space_standard) == 0
    assert clamp_action_safe(action_val=11, action_space=action_space_standard) == 1

def test_yellow_phase_xml_states():
    """Verifies that yellow phases yr (3s) and ry (3s) are configured in modified network plans."""
    import xml.etree.ElementTree as ET
    import os
    
    # We construct a mock minimal XML net file to verify traffic light phases are correctly modified
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <net>
        <tlLogic id="A1" type="static" programID="0" offset="0">
            <phase duration="42" state="GGrrGG"/>
            <phase duration="42" state="yyrryy"/>
            <phase duration="42" state="rrGGrr"/>
            <phase duration="42" state="rryyrr"/>
        </tlLogic>
    </net>
    """
    mock_net_file = "mock_net.xml"
    with open(mock_net_file, "w") as f:
        f.write(xml_content)
        
    try:
        from scale_network import modify_traffic_lights
        modify_traffic_lights(mock_net_file)
        
        # Verify yellow transition phases (phases 1 and 3) are set to duration 3
        tree = ET.parse(mock_net_file)
        root = tree.getroot()
        tl = root.find("tlLogic")
        phases = tl.findall("phase")
        
        assert phases[0].get("duration") == "42"
        assert phases[1].get("duration") == "3", "Yellow transition phase 1 duration was not modified to 3!"
        assert phases[2].get("duration") == "42"
        assert phases[3].get("duration") == "3", "Yellow transition phase 3 duration was not modified to 3!"
    finally:
        if os.path.exists(mock_net_file):
            os.remove(mock_net_file)
