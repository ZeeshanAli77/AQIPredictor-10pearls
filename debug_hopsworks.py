"""
Debug script to check Hopsworks setup
Lists all feature groups, their versions, and all registered models
"""
import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(name: str) -> str:
    """Get secret from environment"""
    return os.getenv(name, "")

try:
    print("🔐 Logging into Hopsworks...")
    project = hopsworks.login(api_key_value=get_secret("HOPSWORKS_API_KEY"))
    print(f"✓ Connected to project: {project.name}\n")
    
    # Check feature store
    print("=" * 60)
    print("📊 FEATURE STORE - Feature Groups")
    print("=" * 60)
    fs = project.get_feature_store()
    
    try:
        fgs = fs.get_feature_groups()
        print(f"Found {len(fgs)} feature groups:\n")
        for fg in fgs:
            print(f"  • {fg.name}")
            print(f"    - Version: {fg.version}")
            print()
    except Exception as e:
        print(f"✗ Error listing feature groups: {e}\n")
    
    # Try to get specific feature group
    print("=" * 60)
    print("🔍 Checking: aqi_features version 7")
    print("=" * 60)
    try:
        fg = fs.get_feature_group("aqi_features", version=7)
        print(f"✓ Found: {fg.name} (v{fg.version})\n")
    except Exception as e:
        print(f"✗ NOT FOUND: {e}\n")
    
    # Check model registry
    print("=" * 60)
    print("🤖 MODEL REGISTRY - Registered Models")
    print("=" * 60)
    mr = project.get_model_registry()
    
    try:
        models = mr.get_models()
        print(f"Found {len(models)} models:\n")
        for model in models:
            print(f"  • {model.name}")
    except Exception as e:
        print(f"✗ Error listing models: {e}\n")
    
    # Check specific models
    print("\n" + "=" * 60)
    print("🔍 Checking specific models")
    print("=" * 60)
    
    model_names = [
        "aqi_predictor_target_aqi_24h",
        "aqi_predictor_target_aqi_48h",
        "aqi_predictor_target_aqi_72h"
    ]
    
    for model_name in model_names:
        try:
            model = mr.get_best_model(model_name, metric="rmse", direction="min")
            print(f"✓ {model_name}")
        except Exception as e:
            print(f"✗ {model_name} - NOT FOUND")

except Exception as e:
    print(f"✗ Fatal error: {e}")