"""
AQI Prediction Inference Module - Standalone predictions with inverse transform.
Can be used by Streamlit, APIs, batch jobs, or any other application.
Models and scalers are loaded from Hopsworks Feature Store.
"""

from __future__ import annotations

import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "")

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


def get_hopsworks_project():
    """Login to Hopsworks project."""
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST
    )


def load_models_and_scalers():
    """Load all 3 trained models and their target scalers from Hopsworks Model Registry.
    
    Returns:
        dict: {
            "24h": {"model": clf_24, "scaler": scaler_24},
            "48h": {"model": clf_48, "scaler": scaler_48},
            "72h": {"model": clf_72, "scaler": scaler_72},
        }
    """
    project = get_hopsworks_project()
    mr = project.get_model_registry()
    
    models_and_scalers = {}
    
    for horizon in ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]:
        key = horizon.replace("target_aqi_", "")  # "24h", "48h", "72h"
        
        # Load best model
        model = mr.get_best_model(f"aqi_predictor_{horizon}", metric="rmse", direction="min")
        model_dir = model.download()
        
        # Load model and scaler
        clf = joblib.load(os.path.join(model_dir, f"aqi_{horizon}_model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, f"aqi_{horizon}_target_scaler.pkl"))
        
        models_and_scalers[key] = {
            "model": clf,
            "scaler": scaler,
        }
    
    return models_and_scalers


def make_predictions(X: pd.DataFrame, models_and_scalers: dict) -> dict:
    """
    Make 24h, 48h, 72h AQI predictions and inverse transform to original AQI scale.
    
    Args:
        X: Feature dataframe with all required columns (FEATURE_COLS)
        models_and_scalers: Output from load_models_and_scalers()
    
    Returns:
        dict: {
            "24h": 70.9 (float),
            "48h": 75.1 (float),
            "72h": 74.3 (float),
        }
    """
    predictions = {}
    
    for horizon in ["24h", "48h", "72h"]:
        model = models_and_scalers[horizon]["model"]
        scaler = models_and_scalers[horizon]["scaler"]
        
        # Make prediction in scaled space
        pred_scaled = float(model.predict(X)[0])
        
        # ✅ INVERSE TRANSFORM to original AQI scale (0-300)
        pred_original = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        
        # Ensure non-negative
        predictions[horizon] = max(0, pred_original)
        
        print(f"[DEBUG] {horizon} prediction: scaled={pred_scaled:.4f} → original={pred_original:.1f}")
    
    return predictions


def get_latest_features_from_store() -> pd.DataFrame:
    """Fetch latest features from Hopsworks Feature Store.
    
    Returns:
        pd.DataFrame: Latest row with all features ready for prediction
    """
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=7)
    df = fg.read()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
    
    # Get latest row
    latest = df.tail(1)
    
    # Verify all features are present
    missing_cols = [col for col in FEATURE_COLS if col not in latest.columns]
    if missing_cols:
        raise ValueError(f"Missing features in latest data: {missing_cols}")
    
    return latest[FEATURE_COLS]


def predict_aqi_forecast(X: pd.DataFrame = None) -> dict:
    """
    Complete end-to-end prediction: Load models, make predictions, inverse transform.
    
    Args:
        X: Optional feature dataframe. If None, fetches latest from Hopsworks.
    
    Returns:
        dict: {
            "24h": 70.9,
            "48h": 75.1,
            "72h": 74.3,
        }
    """
    # Load features if not provided
    if X is None:
        print("[INFO] Fetching latest features from Hopsworks...")
        X = get_latest_features_from_store()
    
    # Ensure correct shape
    if isinstance(X, pd.DataFrame) and len(X) > 1:
        X = X.tail(1)
    
    # Load models and scalers
    print("[INFO] Loading models and scalers from Hopsworks...")
    models_and_scalers = load_models_and_scalers()
    
    # Make predictions
    print("[INFO] Making predictions...")
    predictions = make_predictions(X, models_and_scalers)
    
    return predictions


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("AQI Inference Module - Standalone Test")
    print("="*70)
    
    result = predict_aqi_forecast()
    
    print("\n" + "="*70)
    print("PREDICTIONS (Original AQI Scale 0-300):")
    print("="*70)
    print(f"24h forecast: {result['24h']:.1f} AQI")
    print(f"48h forecast: {result['48h']:.1f} AQI")
    print(f"72h forecast: {result['72h']:.1f} AQI")
    print("="*70)
