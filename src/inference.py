"""
Calculate CORRECT metrics on original AQI scale (0-300).
This is an evaluation-only script — NO retraining, NO model registry registration.
Run once to get accurate metrics and document them.
"""

import os
import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

FEATURE_COLS = [
    "pm25", "pm10", "no2", "co", "o3", "temperature", "humidity",
    "wind_speed", "pressure", "precipitation", "cloud_cover",
    "wind_u", "wind_v", "hour_sin", "hour_cos", "month_sin",
    "month_cos", "dow_sin", "dow_cos", "is_rush_hour", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_change_1h", "aqi_roll_3h", "aqi_roll_6h", "aqi_roll_24h", "aqi_roll_std",
]

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]


def fetch_training_data() -> pd.DataFrame:
    """Fetch features from Hopsworks Feature Store."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")
    
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=7)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
    return df


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features."""
    df = df.sort_values("timestamp").copy()
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
    
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)
    return df


def split_time_series(df: pd.DataFrame, val_fraction: float = 0.15):
    """Split preserving chronological order."""
    split_idx = int(len(df) * (1 - val_fraction))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    return train, val


def load_model_and_scaler(target: str):
    """Load model and target scaler from Hopsworks."""
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    model = mr.get_best_model(f"aqi_predictor_{target}", metric="rmse", direction="min")
    model_dir = model.download()
    
    clf = joblib.load(os.path.join(model_dir, f"aqi_{target}_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, f"aqi_{target}_target_scaler.pkl"))
    
    return clf, scaler


def calculate_metrics_on_original_scale(target: str):
    """Calculate metrics on ORIGINAL AQI scale (0-300), NOT scaled."""
    print(f"\n{'='*70}")
    print(f"Calculating CORRECT metrics for {target}")
    print(f"{'='*70}")
    
    # Fetch and prepare data
    print("Loading data...")
    df = fetch_training_data()
    df = add_time_series_features(df)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    
    _, val_df = split_time_series(df)
    
    X_val = val_df[FEATURE_COLS]
    y_val_original = val_df[target].values  # Original AQI scale
    
    # Load model and scaler
    print(f"Loading model and target scaler for {target}...")
    model, target_scaler = load_model_and_scaler(target)
    
    # Make predictions in scaled space
    print("Making predictions...")
    preds_scaled = model.predict(X_val)
    
    # Inverse transform to original AQI scale
    preds_original = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    
    # Calculate metrics on ORIGINAL scale
    rmse = float(np.sqrt(mean_squared_error(y_val_original, preds_original)))
    mae = float(mean_absolute_error(y_val_original, preds_original))
    r2 = float(r2_score(y_val_original, preds_original))
    
    print(f"\n✓ CORRECT METRICS (Original AQI Scale 0-300):")
    print(f"  RMSE: {rmse:.4f} AQI points")
    print(f"  MAE:  {mae:.4f} AQI points")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\nValidation set statistics:")
    print(f"  y_val range: {y_val_original.min():.1f} - {y_val_original.max():.1f}")
    print(f"  y_val mean: {y_val_original.mean():.1f}")
    print(f"  preds range: {preds_original.min():.1f} - {preds_original.max():.1f}")
    print(f"  preds mean: {preds_original.mean():.1f}")
    
    return {
        "target": target,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "val_samples": len(X_val),
        "calculated_at": datetime.utcnow().isoformat(),
    }


def main():
    print("\n" + "="*70)
    print("AQI METRICS CALCULATION (Original Scale)")
    print("="*70)
    
    all_metrics = {}
    
    for target in TARGET_COLS:
        metrics = calculate_metrics_on_original_scale(target)
        all_metrics[target] = metrics
    
    # Save to file
    output_file = "correct_metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n" + "="*70)
    print("SUMMARY: CORRECT METRICS (Original AQI Scale)")
    print("="*70)
    for target, metrics in all_metrics.items():
        print(f"\n{target}:")
        print(f"  RMSE: {metrics['rmse']} AQI points")
        print(f"  MAE:  {metrics['mae']} AQI points")
        print(f"  R²:   {metrics['r2']}")
    
    print(f"\n✓ Metrics saved to: {output_file}")
    print("\nNOTE: These are the CORRECT metrics on original AQI scale.")
    print("Previous metrics were calculated on StandardScaler normalized space (not representative).")
    print("="*70)


if __name__ == "__main__":
    main()
