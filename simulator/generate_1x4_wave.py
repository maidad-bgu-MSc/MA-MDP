import os
import sys
import subprocess
import xml.etree.ElementTree as ET

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

def run_netgenerate(net_file="wave_1x4.net.xml"):
    """Runs SUMO's netgenerate to build a 4x1 grid corridor network with attached boundary streets."""
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.x-number=4",
        "--grid.y-number=1",
        "--grid.length=200",
        "--grid.attach-length=150",
        "--default-junction-type=traffic_light",
        f"--output-file={net_file}"
    ]
    print(f"Running netgenerate: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running netgenerate:")
        print(result.stderr)
        sys.exit(1)
    print(f"Successfully generated {net_file}!")

def modify_traffic_lights(net_file):
    """Modifies default programs in the generated network:
    1. Standardizes A0, B0, C0, D0 into a stable 2-phase plan with yellow transitions.
    2. Changes boundary junctions to 'priority' to ensure they are not treated as traffic lights.
    3. Removes their tlLogic elements.
    4. Strips 'tl' and 'linkIndex' from connections at boundary junctions to prevent loading errors.
    """
    if not os.path.exists(net_file):
        print(f"Error: {net_file} not found.")
        sys.exit(1)

    tree = ET.parse(net_file)
    root = tree.getroot()

    print(f"Standardizing 2-phase yellow transition durations and clearing boundary traffic lights in {net_file}...")
    
    # Change boundary junctions to priority type so they are not treated as traffic signals
    boundary_ids = {
        "bottom0", "bottom1", "bottom2", "bottom3",
        "top0", "top1", "top2", "top3",
        "left0", "right0"
    }
    for junction in root.findall("junction"):
        j_id = junction.get("id")
        if j_id in boundary_ids:
            junction.set("type", "priority")

    # Remove non-internal tlLogic elements and modify internal ones
    for tl in list(root.findall("tlLogic")):
        tl_id = tl.get("id")
        if tl_id not in {"A0", "B0", "C0", "D0"}:
            root.remove(tl)
        else:
            phases = tl.findall("phase")
            durations = ["42", "3", "42", "3"]
            for idx, dur in enumerate(durations):
                if idx < len(phases):
                    phases[idx].set("duration", dur)
                    
    # Strip 'tl' and 'linkIndex' attributes from connections referencing boundary traffic lights
    for conn in root.findall("connection"):
        tl_attr = conn.get("tl")
        if tl_attr in boundary_ids:
            if "tl" in conn.attrib:
                del conn.attrib["tl"]
            if "linkIndex" in conn.attrib:
                del conn.attrib["linkIndex"]
        
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    print(f"Successfully updated traffic light plans in {net_file}!")

def generate_wave_routes(rou_file="wave_1x4.rou.xml"):
    """Generates horizontal Green Wave platoon flows and stochastic Poisson cross-street flows."""
    routes = {
        "r_eastbound": ["left0A0", "A0B0", "B0C0", "C0D0", "D0right0"],
        "r_westbound": ["right0D0", "D0C0", "C0B0", "B0A0", "A0left0"],
        
        # Vertical cross streets (North-South and South-North)
        "r_ns_0": ["top0A0", "A0bottom0"],
        "r_sn_0": ["bottom0A0", "A0top0"],
        
        "r_ns_1": ["top1B0", "B0bottom1"],
        "r_sn_1": ["bottom1B0", "B0top1"],
        
        "r_ns_2": ["top2C0", "C0bottom2"],
        "r_sn_2": ["bottom2C0", "C0top2"],
        
        "r_ns_3": ["top3D0", "D0bottom3"],
        "r_sn_3": ["bottom3D0", "D0top3"]
    }
    
    with open(rou_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
        
        # Write route definitions
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        # 1. Stochastic Poisson Cross-Street Traffic (lambda = 0.1)
        # MUST BE WRITTEN FIRST because begin="0" and SUMO requires sorted departure times!
        cross_routes = ["r_ns_0", "r_sn_0", "r_ns_1", "r_sn_1", "r_ns_2", "r_sn_2", "r_ns_3", "r_sn_3"]
        for idx, r_id in enumerate(cross_routes):
            f.write(f'    <flow id="cross_flow_{idx}" type="car" begin="0" end="3600" probability="0.1" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            
        f.write("\n")
        
        # 2. Traffic Routing (The Wave): Inject a 30-vehicle platoon over a 10s window every 150s
        for t in range(0, 3600, 150):
            # Eastbound arterial platoon flow
            f.write(f'    <flow id="platoon_eb_{t}" type="car" begin="{t}" end="{t+10}" number="30" route="r_eastbound" departLane="best" departSpeed="max"/>\n')
            # Westbound arterial platoon flow
            f.write(f'    <flow id="platoon_wb_{t}" type="car" begin="{t}" end="{t+10}" number="30" route="r_westbound" departLane="best" departSpeed="max"/>\n')
            
        f.write('</routes>\n')
    print(f"Successfully generated wave routes: {rou_file}")

def build_1x4_scenario():
    """Autonomously compiles and builds the 1x4 arterial wave SUMO network and route flows."""
    setup_sumo_env()
    run_netgenerate()
    modify_traffic_lights("wave_1x4.net.xml")
    generate_wave_routes("wave_1x4.rou.xml")
    print("1x4 Wave Corridor network completely compiled under wave_1x4.net.xml and wave_1x4.rou.xml!\n")
    return "wave_1x4.net.xml", "wave_1x4.rou.xml"

if __name__ == "__main__":
    build_1x4_scenario()
