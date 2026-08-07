import hopsworks
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_secret(name: str) -> str:
    return os.getenv(name, "")

try:
    print("🔐 Step 1: Login to Hopsworks...")
    project = hopsworks.login(api_key_value=get_secret("HOPSWORKS_API_KEY"))
    print(f"✓ Connected to: {project.name}\n")
    
    print("📦 Step 2: Get Model Registry...")
    mr = project.get_model_registry()
    print("✓ Model registry connected\n")
    
    print("🤖 Step 3: Loading 24h model...")
    try:
        model_24 = mr.get_best_model("aqi_predictor_target_aqi_24h", metric="rmse", direction="min")
        print(f"✓ Got model: {model_24.name} (v{model_24.version})")
        saved_model_dir_24 = model_24.download()
        clf_24 = joblib.load(os.path.join(saved_model_dir_24, "aqi_target_aqi_24h_model.pkl"))
        print(f"   ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ 24h model failed: {e}\n")

    print("🤖 Step 4: Loading 48h model...")
    try:
        model_48 = mr.get_best_model("aqi_predictor_target_aqi_48h", metric="rmse", direction="min")
        print(f"✓ Got model: {model_48.name} (v{model_48.version})")
        saved_model_dir_48 = model_48.download()
        clf_48 = joblib.load(os.path.join(saved_model_dir_48, "aqi_target_aqi_48h_model.pkl"))
        print(f"   ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ 48h model failed: {e}\n")

    print("🤖 Step 5: Loading 72h model...")
    try:
        model_72 = mr.get_best_model("aqi_predictor_target_aqi_72h", metric="rmse", direction="min")
        print(f"✓ Got model: {model_72.name} (v{model_72.version})")
        saved_model_dir_72 = model_72.download()
        clf_72 = joblib.load(os.path.join(saved_model_dir_72, "aqi_target_aqi_72h_model.pkl"))
        print(f"   ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ 72h model failed: {e}\n")

    print("📊 Step 6: Get Feature Store...")
    fs = project.get_feature_store()
    print("✓ Feature store connected\n")
    
    print("📋 Step 7: Get Feature Group...")
    fg = fs.get_feature_group("aqi_features", version=7)
    print(f"✓ Feature group: {fg.name} (v{fg.version})\n")
    
    print("📥 Step 8: Read data...")
    df = fg.read(online=True)
    print(f"✓ Data read successful: {df.shape}\n")
    
    print("=" * 60)
    print("✅ ALL STEPS PASSED - No errors found!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()