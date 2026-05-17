import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from scale_network import setup_sumo_env, modify_traffic_lights

def generate_bottleneck_xml_files():
    """Generates intermediate bottleneck.nod.xml and bottleneck.edg.xml files."""
    # Nodes defining coordinated 4-way junctions
    nodes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="A" x="0.0" y="0.0" type="traffic_light"/>
    <node id="B" x="200.0" y="0.0" type="traffic_light"/>
    <node id="W" x="-150.0" y="0.0" type="priority"/>
    <node id="E" x="350.0" y="0.0" type="priority"/>
    <node id="NA" x="0.0" y="150.0" type="priority"/>
    <node id="SA" x="0.0" y="-150.0" type="priority"/>
    <node id="NB" x="200.0" y="150.0" type="priority"/>
    <node id="SB" x="200.0" y="-150.0" type="priority"/>
</nodes>
"""
    with open("bottleneck.nod.xml", "w") as f:
        f.write(nodes_xml)

    # Edges defining capacity drop (3 lanes to 1 lane between A and B)
    edges_xml = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Main arterial corridor -->
    <edge id="W_to_A" from="W" to="A" numLanes="3" speed="13.89"/>
    <edge id="A_to_W" from="A" to="W" numLanes="3" speed="13.89"/>
    
    <!-- Central bottleneck drop to 1 lane -->
    <edge id="A_to_B" from="A" to="B" numLanes="1" speed="13.89"/>
    <edge id="B_to_A" from="B" to="A" numLanes="1" speed="13.89"/>
    
    <edge id="B_to_E" from="B" to="E" numLanes="3" speed="13.89"/>
    <edge id="E_to_B" from="E" to="B" numLanes="3" speed="13.89"/>

    <!-- Intersecting cross-corridors -->
    <edge id="NA_to_A" from="NA" to="A" numLanes="2" speed="13.89"/>
    <edge id="A_to_NA" from="A" to="NA" numLanes="2" speed="13.89"/>
    <edge id="SA_to_A" from="SA" to="A" numLanes="2" speed="13.89"/>
    <edge id="A_to_SA" from="A" to="SA" numLanes="2" speed="13.89"/>
    
    <edge id="NB_to_B" from="NB" to="B" numLanes="2" speed="13.89"/>
    <edge id="B_to_NB" from="B" to="NB" numLanes="2" speed="13.89"/>
    <edge id="SB_to_B" from="SB" to="B" numLanes="2" speed="13.89"/>
    <edge id="B_to_SB" from="B" to="SB" numLanes="2" speed="13.89"/>
</edges>
"""
    with open("bottleneck.edg.xml", "w") as f:
        f.write(edges_xml)

def run_netconvert():
    """Compiles custom bottleneck node/edge files into bottleneck.net.xml."""
    cmd = [
        "netconvert",
        "--node-files=bottleneck.nod.xml",
        "--edge-files=bottleneck.edg.xml",
        "--output-file=bottleneck.net.xml"
    ]
    print(f"Running netconvert for bottleneck: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running netconvert:")
        print(result.stderr)
        sys.exit(1)
    print("Successfully compiled bottleneck.net.xml!")

def generate_bottleneck_routes(rou_file):
    """Generates route patterns through the central bottleneck corridor."""
    routes = {
        "r_eastbound": ["W_to_A", "A_to_B", "B_to_E"],
        "r_westbound": ["E_to_B", "B_to_A", "A_to_W"],
        "r_cross_A": ["NA_to_A", "A_to_SA"],
        "r_cross_B": ["NB_to_B", "B_to_SB"]
    }
    
    with open(rou_file, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89"/>\n\n')
        
        for r_id, r_edges in routes.items():
            f.write(f'    <route id="{r_id}" edges="{" ".join(r_edges)}"/>\n')
            
        f.write("\n")
        
        # High volume of traffic passing eastbound/westbound directly through bottleneck
        # Cross traffic runs at a moderate Poisson rate
        for r_id in routes.keys():
            if r_id in ["r_eastbound", "r_westbound"]:
                # High density demand driving saturation on bottleneck edge
                f.write(f'    <flow id="bottleneck_flow_{r_id}" type="car" begin="0" end="3600" period="exp(0.04)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
            else:
                f.write(f'    <flow id="cross_flow_{r_id}" type="car" begin="0" end="3600" period="exp(0.1)" route="{r_id}" departLane="best" departSpeed="max"/>\n')
                
        f.write('</routes>\n')
    print(f"Successfully generated bottleneck route flows: {rou_file}")

def build_bottleneck_scenario():
    """Builds Scenario 3 (Capacity Drops)."""
    setup_sumo_env()
    generate_bottleneck_xml_files()
    run_netconvert()
    
    # Standardize yellow phases in the custom bottleneck network
    modify_traffic_lights("bottleneck.net.xml")
    
    generate_bottleneck_routes("bottleneck.rou.xml")
    
    # Clean up intermediate files
    for temp_file in ["bottleneck.nod.xml", "bottleneck.edg.xml"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    print("Scenario 3 (Bottleneck) successfully compiled under bottleneck.net.xml and bottleneck.rou.xml!\n")
    return "bottleneck.net.xml", "bottleneck.rou.xml"

if __name__ == "__main__":
    build_bottleneck_scenario()
