import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from scale_network import setup_sumo_env, modify_traffic_lights

def run_netgenerate_spillback(size=2):
    """Runs netgenerate to dynamically create the short-corridor grid network."""
    net_file = f"grid_{size}x{size}_spillback.net.xml"
    cmd = [
        "netgenerate",
        "--grid",
        f"--grid.number={size}",
        "--grid.length=50",
        "--default-junction-type=traffic_light",
        f"--output-file={net_file}"
    ]
    print(f"Generating spillback network: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running netgenerate spillback:")
        print(result.stderr)
        sys.exit(1)
    print(f"Successfully generated {net_file} with 50m corridors.")
    return net_file

def generate_spillback_routes(size, rou_file):
    """Generates Horizontal and Vertical Corridor Route flows with near-saturation Poisson density."""
    cols = [chr(65 + i) for i in range(size)]
    routes = {}
    
    # 1. Generate Horizontal Corridor Routes
    for r in range(size):
        # Eastbound
        right_edges = []
        for c in range(size - 1):
            right_edges.append(f"{cols[c]}{r}{cols[c+1]}{r}")
        routes[f"route_h_right_{r}"] = right_edges
        
        # Westbound
        left_edges = []
        for c in range(size - 1 - 1, -1, -1):
            left_edges.append(f"{cols[c+1]}{r}{cols[c]}{r}")
        routes[f"route_h_left_{r}"] = left_edges

    # 2. Generate Vertical Corridor Routes
    for c in range(size):
        # Northbound
        up_edges = []
        for r in range(size - 1):
            up_edges.append(f"{cols[c]}{r}{cols[c]}{r+1}")
        routes[f"route_v_up_{c}"] = up_edges
        
        # Southbound
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
        
        # Near-saturation Poisson flow rates (exp(0.015)) causing rapid queues
        flow_idx = 0
        for r_id in routes.keys():
            f.write(f'    <flow id="flow_{flow_idx}" type="car" begin="0" end="3600" period="exp(0.015)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            flow_idx += 1
            
        f.write('</routes>\n')
    print(f"Successfully generated spillback routes: {rou_file}")

def build_spillback_scenario(size=2):
    """Builds Scenario 4 (Spillback & Saturation)."""
    setup_sumo_env()
    net_file = run_netgenerate_spillback(size)
    modify_traffic_lights(net_file)
    
    rou_file = f"grid_{size}x{size}_spillback.rou.xml"
    generate_spillback_routes(size, rou_file)
    
    print(f"Scenario 4 (Spillback) successfully compiled under {net_file} and {rou_file}!\n")
    return net_file, rou_file

if __name__ == "__main__":
    build_spillback_scenario(size=2)
