"""
Inference helpers for AQI prediction.
FIXED: Now properly handles models with StandardScaler in pipelines.
The scaler is automatically applied through the loaded pipeline.
UPDATED: Uses latest model version instead of best RMSE (avoids overfitting).
"""
from __future__ import annotations
import os
from typing import Dict
import joblib
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

FEATURE_COLS = [
    "pm25",
    "pm10",
    "no2",
    "co",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "precipitation",
    "cloud_cover",
    "wind_u",
    "wind_v",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_rush_hour",
    "is_weekend",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_24h",
    "aqi_change_1h",
    "aqi_roll_3h",
    "aqi_roll_6h",
    "aqi_roll_24h",
    "aqi_roll_std",
]


def load_latest_features() -> pd.DataFrame:
    """Load latest features from Hopsworks Feature Store with lag computations."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")
    
    import tempfile
    
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    # Auto-select latest version and read from OFFLINE store
    fg = fs.get_feature_group("aqi_features")
    
    # Force fresh read by using a temp directory and setting cache=False
    temp_dir = tempfile.mkdtemp()
    df = fg.read(cache=False)  # ✅ CRITICAL: Bypass cache
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
    
    # DEBUG: Print what was fetched
    print(f"\n[DEBUG] Offline store fetch (CACHE BYPASSED):")
    print(f"  Total rows: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Latest AQI value: {df['aqi'].iloc[-1] if len(df) > 0 else 'N/A'}")
    print(f"[DEBUG] Latest timestamp: {df['timestamp'].max()}")
    
    if len(df) < 25:
        raise RuntimeError("Not enough history to compute lag features.")
    
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_6h"] = df["aqi"].shift(6)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_change_1h"] = df["aqi"] - df["aqi_lag_1h"]
    df["aqi_roll_3h"] = df["aqi"].rolling(3).mean()
    df["aqi_roll_6h"] = df["aqi"].rolling(6).mean()
    df["aqi_roll_24h"] = df["aqi"].rolling(24).mean()
    df["aqi_roll_std"] = df["aqi"].rolling(6).std()
    
    latest = df.tail(1)
    X = latest[FEATURE_COLS]
    
    if X.isna().any(axis=None):
        valid = df[FEATURE_COLS].dropna()
        if valid.empty:
            raise RuntimeError("Latest features contain NaNs; check historical data.")
        X = valid.tail(1)
    
    return X


def load_best_model(target: str) -> object:
    """Load the LATEST model version for a given target from Hopsworks Model Registry."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")
    
    import tempfile
    
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    model_name = f"aqi_predictor_{target}"
    
    # ✅ FIXED: Get LATEST version by listing all versions and picking highest
    model = None
    try:
        print(f"[DEBUG] Querying all versions for {model_name}...")
        # List all model versions
        all_models = mr.list_models(model_name)
        print(f"[DEBUG] Found {len(all_models)} versions")
        
        if not all_models:
            raise Exception(f"No versions found for {model_name}")
        
        # Get version numbers and find the maximum
        versions = []
        for m in all_models:
            try:
                v = int(m.version)
                versions.append((v, m))
            except (ValueError, TypeError):
                continue
        
        if not versions:
            raise Exception(f"Could not parse versions for {model_name}")
        
        # Sort and get latest
        versions.sort(key=lambda x: x[0])
        latest_version_num, _ = versions[-1]
        
        model = mr.get_model(model_name, version=str(latest_version_num))
        print(f"[✓] Loading {model_name} v{model.version} (LATEST)")
        
    except Exception as e:
        print(f"[⚠] Failed to get latest version ({e}), falling back to best RMSE")
        model = mr.get_best_model(model_name, metric="rmse", direction="min")
        print(f"[!] Loaded {model_name} v{model.version} (best RMSE - fallback)")
    
    # Force fresh download to bypass cache
    temp_dir = tempfile.mkdtemp()
    model_dir = model.download(local_path=temp_dir)
    
    print(f"[DEBUG] Downloaded v{model.version} to {model_dir}")
    
    # Load the pipeline (includes scaler + model)
    model_pkl_path = os.path.join(model_dir, f"aqi_{target}_model.pkl")
    return joblib.load(model_pkl_path)


def predict_latest() -> Dict[str, float]:
    """
    Predict AQI for 24h, 48h, and 72h ahead.
    Scaling is automatically handled by the loaded pipeline.
    Uses LATEST model versions for all targets.
    """
    X = load_latest_features()
    preds = {}
    
    for target in ("target_aqi_24h", "target_aqi_48h", "target_aqi_72h"):
        model = load_best_model(target)
        # model.predict() automatically applies scaler from pipeline
        preds[target] = float(model.predict(X)[0])
    
    return preds


if __name__ == "__main__":
    print(predict_latest())
