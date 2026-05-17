import os
import sys
import xml.etree.ElementTree as ET
from scale_network import setup_sumo_env, run_netgenerate, modify_traffic_lights

def generate_platoon_routes(size, rou_file):
    """Generates a route file with 40-vehicle platoons injected every 300 seconds on a single corridor."""
    cols = [chr(65 + i) for i in range(size)]
    routes = {}
    
    # 1. Generate Horizontal Corridor Routes (consistent with scale_network.py)
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
        
        # Write route paths
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        # Designate route_h_right_0 as the "Platoon Corridor"
        platoon_route = "route_h_right_0"
        veh_idx = 0
        
        # All other routes have a low, steady cross-traffic Poisson flow
        flow_idx = 0
        for r_id in routes.keys():
            if r_id != platoon_route:
                f.write(f'    <flow id="cross_flow_{flow_idx}" type="car" begin="0" end="3600" period="exp(0.3)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
                flow_idx += 1
                
        f.write("\n")
        
        # Inject tightly packed platoon of 40 vehicles every 300s
        for platoon_start in range(0, 3600, 300):
            f.write(f'    <!-- Platoon burst starting at {platoon_start}s -->\n')
            for gap in range(40):
                depart_time = platoon_start + gap
                f.write(f'    <vehicle id="platoon_veh_{veh_idx}" type="car" depart="{depart_time}" route="{platoon_route}" departLane="best" departSpeed="max"/>\n')
                veh_idx += 1
                
        f.write('</routes>\n')
    print(f"Successfully generated platoon routes file: {rou_file}")

def build_platoon_scenario(size=2):
    """Builds the network and route files for Scenario 1."""
    setup_sumo_env()
    net_file = f"grid_{size}x{size}_platoon.net.xml"
    
    # 1. Generate grid network using scale_network runner
    # We rename or run netgenerate output directly
    import shutil
    net_base = run_netgenerate(size)
    shutil.copy(net_base, net_file)
    
    # 2. Standardize yellow phases in the custom network
    modify_traffic_lights(net_file)
    
    # 3. Generate the platoon routes
    rou_file = f"grid_{size}x{size}_platoon.rou.xml"
    generate_platoon_routes(size, rou_file)
    print(f"Scenario 1 (Platoon) successfully compiled under {net_file} and {rou_file}!\n")
    return net_file, rou_file

if __name__ == "__main__":
    build_platoon_scenario(size=2)
