import os
import shutil
import xml.etree.ElementTree as ET
from scale_network import setup_sumo_env, run_netgenerate, modify_traffic_lights, generate_routes

def test_net_generation():
    """Asserts that SUMO network and route XML files are physically created, parsed, and validated."""
    net_file = "grid_2x2.net.xml"
    rou_file = "grid_2x2.rou.xml"
    
    net_backup = net_file + ".bak"
    rou_backup = rou_file + ".bak"
    
    # 1. Back up existing files if they exist to keep workspace pristine
    if os.path.exists(net_file):
        shutil.copy(net_file, net_backup)
    if os.path.exists(rou_file):
        shutil.copy(rou_file, rou_backup)
        
    try:
        # Run PATH injections
        setup_sumo_env()
        
        # 2. Test netgenerate invocation
        generated_net = run_netgenerate(size=2)
        assert generated_net == net_file
        assert os.path.exists(net_file), "Network XML file was not created!"
        assert os.path.getsize(net_file) > 100, "Network XML file is too small or empty!"
        
        # 3. Test XML parsing and modification
        modify_traffic_lights(net_file)
        tree = ET.parse(net_file)
        root = tree.getroot()
        tl_logics = root.findall("tlLogic")
        assert len(tl_logics) > 0, "Failed to parse modified XML or no traffic lights found!"
        
        # 4. Test route generation
        generate_routes(size=2, rou_file=rou_file)
        assert os.path.exists(rou_file), "Route XML file was not created!"
        assert os.path.getsize(rou_file) > 100, "Route XML file is too small or empty!"
        
    finally:
        # 5. Clean up temporary test files
        if os.path.exists(net_file):
            os.remove(net_file)
        if os.path.exists(rou_file):
            os.remove(rou_file)
            
        # 6. Restore original backups
        if os.path.exists(net_backup):
            shutil.move(net_backup, net_file)
        if os.path.exists(rou_backup):
            shutil.move(rou_backup, rou_file)
