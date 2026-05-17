import os
import sys
import subprocess
import argparse
import xml.etree.ElementTree as ET

def setup_sumo_env():
    """Locates sumo / netgenerate executables and sets SUMO_HOME programmatically."""
    try:
        import sumo
        sumo_home = os.path.abspath(os.path.dirname(sumo.__file__))
        os.environ["SUMO_HOME"] = sumo_home
        print(f"SUMO_HOME set programmatically to: {sumo_home}")
    except ImportError:
        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            print("Warning: SUMO_HOME environment variable is not set!")
            
    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")
    
    paths = []
    if sumo_home:
        paths.append(os.path.join(sumo_home, "bin"))
    paths.append(scripts_dir)
    
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + current_path

def run_netgenerate(size):
    """Runs netgenerate to dynamically create the KxK traffic light grid network."""
    net_file = f"grid_{size}x{size}.net.xml"
    cmd = [
        "netgenerate",
        "--grid",
        f"--grid.number={size}",
        "--grid.length=150",
        "--default-junction-type=traffic_light",
        f"--output-file={net_file}"
    ]
    print(f"Generating {size}x{size} Grid: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running netgenerate for size {size}:")
        print(result.stderr)
        sys.exit(1)
    print(f"Successfully generated {net_file}")
    return net_file

def modify_traffic_lights(net_file):
    """Modifies the default multi-phase programs in the generated network into a stable 2-phase plan with yellow transitions."""
    if not os.path.exists(net_file):
        print(f"Error: {net_file} not found.")
        sys.exit(1)

    tree = ET.parse(net_file)
    root = tree.getroot()

    print(f"Standardizing 2-phase yellow transition durations in {net_file}...")
    for tl in root.findall("tlLogic"):
        phases = tl.findall("phase")
        # Standardize phases 0, 1, 2, 3 to Gr (42s), yr (3s), rG (42s), ry (3s) while preserving computed states
        durations = ["42", "3", "42", "3"]
        for idx, dur in enumerate(durations):
            if idx < len(phases):
                phases[idx].set("duration", dur)
        
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    print(f"Successfully updated traffic light plans in {net_file}!")

def generate_routes(size, rou_file):
    """Generates horizontal and vertical corridor route flows at a consistent Poisson rate."""
    cols = [chr(65 + i) for i in range(size)]
    rows = [str(i) for i in range(size)]
    routes = {}
    
    # 1. Generate Horizontal Corridor Routes
    for r in range(size):
        # Going right (Eastbound)
        right_edges = []
        for c in range(size - 1):
            right_edges.append(f"{cols[c]}{r}{cols[c+1]}{r}")
        routes[f"route_h_right_{r}"] = right_edges
        
        # Going left (Westbound)
        left_edges = []
        for c in range(size - 1 - 1, -1, -1):
            left_edges.append(f"{cols[c+1]}{r}{cols[c]}{r}")
        routes[f"route_h_left_{r}"] = left_edges

    # 2. Generate Vertical Corridor Routes
    for c in range(size):
        # Going up (Northbound)
        up_edges = []
        for r in range(size - 1):
            up_edges.append(f"{cols[c]}{r}{cols[c]}{r+1}")
        routes[f"route_v_up_{c}"] = up_edges
        
        # Going down (Southbound)
        down_edges = []
        for r in range(size - 1 - 1, -1, -1):
            down_edges.append(f"{cols[c]}{r+1}{cols[c]}{r}")
        routes[f"route_v_down_{c}"] = down_edges

    # Write out to route file
    with open(rou_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
        
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        # Consistently scaled Poisson traffic flows
        flow_idx = 0
        for r_id in routes.keys():
            f.write(f'    <flow id="flow_{flow_idx}" type="car" begin="0" end="3600" period="exp(0.05)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            flow_idx += 1
            
        f.write('</routes>\n')
    print(f"Successfully generated {rou_file} with consistent Poisson flow density!")

def build_network_size(size):
    setup_sumo_env()
    net_file = run_netgenerate(size)
    modify_traffic_lights(net_file)
    
    rou_file = f"grid_{size}x{size}.rou.xml"
    generate_routes(size, rou_file)
    print(f"Grid size {size}x{size} network completely built!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale ATLC Traffic Light grid networks.")
    parser.add_argument("--size", type=int, default=2, choices=[2, 3, 4, 5], help="Grid number size K for KxK junctions.")
    args = parser.parse_args()
    
    build_network_size(args.size)
