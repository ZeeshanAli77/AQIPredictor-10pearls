import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

try:
    print("Testing feature group read...\n")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    print(f"✓ Connected to project: {project.name}\n")
    
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=7)
    
    # Test offline read
    print("1️⃣ Reading OFFLINE...")
    try:
        df_offline = fg.read()
        print(f"✓ Offline read successful: {df_offline.shape}\n")
    except Exception as e:
        print(f"✗ Offline read failed: {e}\n")
    
    # Test online read
    print("2️⃣ Reading ONLINE...")
    try:
        df_online = fg.read(online=True)
        print(f"✓ Online read successful: {df_online.shape}\n")
    except Exception as e:
        print(f"✗ Online read failed: {e}\n")
        
except Exception as e:
    print(f"✗ Connection failed: {e}")