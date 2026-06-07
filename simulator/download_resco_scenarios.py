import os
import urllib.request
import urllib.error
import zipfile
import io

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Success!")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def download_resco_data():
    dest_dir = os.path.join("simulator", "resco_environments")
    os.makedirs(dest_dir, exist_ok=True)
    
    # Try master branch first, then main branch
    branches = ["master", "main"]
    
    # 1. Download cologne3
    cologne_dir = os.path.join(dest_dir, "cologne3")
    os.makedirs(cologne_dir, exist_ok=True)
    for file in ["cologne3.net.xml", "cologne3.rou.xml"]:
        dest_file = os.path.join(cologne_dir, file)
        if os.path.exists(dest_file):
            print(f"{dest_file} already exists, skipping.")
            continue
        success = False
        for branch in branches:
            url = f"https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/{branch}/resco_benchmark/environments/cologne3/{file}"
            if download_file(url, dest_file):
                success = True
                break
        if not success:
            raise RuntimeError(f"Failed to download {file} from master/main branches.")
            
    # 2. Download grid4x4
    grid_dir = os.path.join(dest_dir, "grid4x4")
    os.makedirs(grid_dir, exist_ok=True)
    
    # net file
    net_file = os.path.join(grid_dir, "grid4x4.net.xml")
    if not os.path.exists(net_file):
        success = False
        for branch in branches:
            url = f"https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/{branch}/resco_benchmark/environments/grid4x4/grid4x4.net.xml"
            if download_file(url, net_file):
                success = True
                break
        if not success:
            raise RuntimeError("Failed to download grid4x4.net.xml")
            
    # rou file via zip
    rou_file = os.path.join(grid_dir, "grid4x4.rou.xml")
    if not os.path.exists(rou_file):
        success = False
        for branch in branches:
            url = f"https://raw.githubusercontent.com/Pi-Star-Lab/RESCO/{branch}/resco_benchmark/environments/grid4x4/grid4x4.zip"
            print(f"Downloading zip from {url}...")
            try:
                # Use urllib to get request and read the bytes
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                data = urllib.request.urlopen(req).read()
                z = zipfile.ZipFile(io.BytesIO(data))
                # Extract specific route file content and write it
                content = z.read("grid4x4_1.rou.xml")
                with open(rou_file, "wb") as f:
                    f.write(content)
                print("Successfully extracted grid4x4_1.rou.xml to grid4x4.rou.xml!")
                success = True
                break
            except Exception as e:
                print(f"Failed: {e}")
        if not success:
            raise RuntimeError("Failed to download/extract grid4x4.rou.xml from zip.")


if __name__ == "__main__":
    download_resco_data()
