"""
Inference helpers for AQI prediction.
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
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("aqi_features", version=1)
    df = fg.read(online=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
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
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    model = mr.get_best_model(f"aqi_predictor_{target}", metric="rmse", direction="min")
    model_dir = model.download()
    return joblib.load(os.path.join(model_dir, f"aqi_{target}_model.pkl"))


def predict_latest() -> Dict[str, float]:
    X = load_latest_features()
    preds = {}
    for target in ("target_aqi_24h", "target_aqi_48h", "target_aqi_72h"):
        model = load_best_model(target)
        preds[target] = float(model.predict(X)[0])
    return preds


if __name__ == "__main__":
    print(predict_latest())
