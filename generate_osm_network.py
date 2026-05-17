import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from scale_network import setup_sumo_env, modify_traffic_lights

def generate_osm_xml_files():
    """Generates the intermediate osm.nod.xml and osm.edg.xml files."""
    # Nodes file defining T-junctions and 4-way intersections
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="A" x="0.0" y="0.0" type="traffic_light"/>
    <node id="B" x="200.0" y="0.0" type="traffic_light"/>
    <node id="C" x="400.0" y="0.0" type="traffic_light"/>
    <node id="W" x="-150.0" y="0.0" type="priority"/>
    <node id="E" x="550.0" y="0.0" type="priority"/>
    <node id="NA" x="0.0" y="150.0" type="priority"/>
    <node id="NB" x="200.0" y="150.0" type="priority"/>
    <node id="SB" x="200.0" y="-150.0" type="priority"/>
    <node id="SC" x="400.0" y="-150.0" type="priority"/>
</nodes>
"""
    with open("osm.nod.xml", "w") as f:
        f.write(nodes_xml)

    # Edges file defining 3-lane main arterial and 1-2 lane peripheral roads
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Main corridor edges -->
    <edge id="W_to_A" from="W" to="A" numLanes="3" speed="13.89"/>
    <edge id="A_to_W" from="A" to="W" numLanes="3" speed="13.89"/>
    
    <edge id="A_to_B" from="A" to="B" numLanes="3" speed="13.89"/>
    <edge id="B_to_A" from="B" to="A" numLanes="3" speed="13.89"/>
    
    <edge id="B_to_C" from="B" to="C" numLanes="3" speed="13.89"/>
    <edge id="C_to_B" from="C" to="B" numLanes="3" speed="13.89"/>
    
    <edge id="C_to_E" from="C" to="E" numLanes="3" speed="13.89"/>
    <edge id="E_to_C" from="E" to="C" numLanes="3" speed="13.89"/>

    <!-- Peripheral intersecting edges -->
    <edge id="NA_to_A" from="NA" to="A" numLanes="1" speed="13.89"/>
    <edge id="A_to_NA" from="A" to="NA" numLanes="1" speed="13.89"/>
    
    <edge id="NB_to_B" from="NB" to="B" numLanes="2" speed="13.89"/>
    <edge id="B_to_NB" from="B" to="NB" numLanes="2" speed="13.89"/>
    
    <edge id="SB_to_B" from="SB" to="B" numLanes="2" speed="13.89"/>
    <edge id="B_to_SB" from="B" to="SB" numLanes="2" speed="13.89"/>
    
    <edge id="SC_to_C" from="SC" to="C" numLanes="1" speed="13.89"/>
    <edge id="C_to_SC" from="C" to="SC" numLanes="1" speed="13.89"/>
</edges>
"""
    with open("osm.edg.xml", "w") as f:
        f.write(edges_xml)

def run_netconvert():
    """Runs netconvert to compile custom node and edge files into osm.net.xml."""
    cmd = [
        "netconvert",
        "--node-files=osm.nod.xml",
        "--edge-files=osm.edg.xml",
        "--output-file=osm.net.xml"
    ]
    print(f"Running netconvert: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running netconvert:")
        print(result.stderr)
        sys.exit(1)
    print("Successfully compiled osm.net.xml using netconvert!")

def generate_osm_routes(rou_file):
    """Generates the routes and Poisson flows mapping the asymmetric grid."""
    routes = {
        "r_eastbound": ["W_to_A", "A_to_B", "B_to_C", "C_to_E"],
        "r_westbound": ["E_to_C", "C_to_B", "B_to_A", "B_to_A", "A_to_W"], # note correct edge sequence
        "r_NA_A_W": ["NA_to_A", "A_to_W"],
        "r_W_A_NA": ["W_to_A", "A_to_NA"],
        "r_NB_B_SB": ["NB_to_B", "B_to_SB"],
        "r_SB_B_NB": ["SB_to_B", "B_to_NB"],
        "r_SC_C_E": ["SC_to_C", "C_to_E"],
        "r_E_C_SC": ["E_to_C", "C_to_SC"]
    }
    
    # Fix the edge list for westwards to ensure exact matching connected transitions
    routes["r_westbound"] = ["E_to_C", "C_to_B", "B_to_A", "A_to_W"]

    with open(rou_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
        
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        # Poisson traffic flows
        flow_idx = 0
        for r_id in routes.keys():
            f.write(f'    <flow id="flow_{flow_idx}" type="car" begin="0" end="3600" period="exp(0.05)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            flow_idx += 1
            
        f.write('</routes>\n')
    print(f"Successfully generated OSM asymmetric route flows: {rou_file}")

def build_osm_scenario():
    """Builds Scenario 2 (Asymmetric Topologies)."""
    setup_sumo_env()
    generate_osm_xml_files()
    run_netconvert()
    
    # Standardize yellow phases in the custom network
    modify_traffic_lights("osm.net.xml")
    
    generate_osm_routes("osm.rou.xml")
    
    # Clean up intermediate xml files to keep workspace clean
    for temp_file in ["osm.nod.xml", "osm.edg.xml"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("Scenario 2 (OSM Asymmetric) successfully compiled under osm.net.xml and osm.rou.xml!\n")
    return "osm.net.xml", "osm.rou.xml"

if __name__ == "__main__":
    build_osm_scenario()
