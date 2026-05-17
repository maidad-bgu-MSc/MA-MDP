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
        print(f"SUMO_HOME set programmatically to: {sumo_home}")
    except ImportError:
        print("Warning: 'sumo' Python package not found, relying on system PATH.")
        sumo_home = os.environ.get("SUMO_HOME")

    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")
    
    paths = []
    if sumo_home:
        paths.append(os.path.join(sumo_home, "bin"))
    paths.append(scripts_dir)
    
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + current_path
    print(f"Added paths to PATH: {paths}")

def run_netgenerate():
    """Runs netgenerate to create the 2x2 traffic light grid network."""
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.number=2",
        "--grid.length=150",
        "--default-junction-type=traffic_light",
        "--output-file=grid.net.xml"
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running netgenerate:")
        print(result.stderr)
        sys.exit(1)
    print("Successfully generated grid.net.xml!")

def modify_traffic_lights():
    """Programmatically modifies the traffic light logic in grid.net.xml to have 2 active phases and yellow transitions."""
    if not os.path.exists("grid.net.xml"):
        print("Error: grid.net.xml not found.")
        sys.exit(1)

    tree = ET.parse("grid.net.xml")
    root = tree.getroot()

    print("Standardizing traffic light durations in grid.net.xml...")
    for tl in root.findall("tlLogic"):
        phases = tl.findall("phase")
        durations = ["42", "3", "42", "3"]
        for idx, dur in enumerate(durations):
            if idx < len(phases):
                phases[idx].set("duration", dur)
        
    tree.write("grid.net.xml", encoding="UTF-8", xml_declaration=True)
    print("Successfully updated traffic light program inside grid.net.xml!")

def generate_routes():
    """Generates the route file grid.rou.xml with 8 contiguous intersecting routes and Poisson flows."""
    routes = {
        "r_A1A0_A0B0": ["A1A0", "A0B0"],
        "r_B0A0_A0A1": ["B0A0", "A0A1"],
        "r_B1A1_A1A0": ["B1A1", "A1A0"],
        "r_A0A1_A1B1": ["A0A1", "A1B1"],
        "r_B1B0_B0A0": ["B1B0", "B0A0"],
        "r_A0B0_B0B1": ["A0B0", "B0B1"],
        "r_B0B1_B1A1": ["B0B1", "B1A1"],
        "r_A1B1_B1B0": ["A1B1", "B1B0"]
    }

    with open("grid.rou.xml", "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
        
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        flow_idx = 0
        for r_id in routes.keys():
            f.write(f'    <flow id="flow_{flow_idx}" type="car" begin="0" end="3600" period="exp(0.05)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            flow_idx += 1
            
        f.write('</routes>\n')
    print("Successfully generated grid.rou.xml with Poisson flows!")

if __name__ == "__main__":
    setup_sumo_env()
    run_netgenerate()
    modify_traffic_lights()
    generate_routes()
