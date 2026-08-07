import hopsworks

try:
    project = hopsworks.login()
    print(f"✓ Connected to project: {project.name}")
except Exception as e:
    print(f"✗ Connection failed: {e}")