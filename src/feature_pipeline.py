"""
Feature Pipeline: Fetch, compute, and store AQI features for Islamabad/Rawalpindi.
Runs every hour via GitHub Actions.
Uses Open-Meteo Air Quality API (matches training data backfill source).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import hopsworks
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "features.csv")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
CITY = CONFIG["city"]
LAT, LON = CITY["latitude"], CITY["longitude"]

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")


def fetch_aqi_data(lat: float, lon: float) -> dict:
    """Fetch current air quality data from Open-Meteo Air Quality API (no API key needed)."""
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,dust,european_aqi"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["current"]


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather from Open-Meteo (no API key needed)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,surface_pressure,precipitation,cloud_cover"
        "&forecast_days=1"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()["current"]


def extract_pollutants(aqi_data: dict) -> dict:
    """Extract pollutant readings from Open-Meteo Air Quality response."""
    def to_float(value) -> float:
        if value in (None, "-", ""):
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    # DEBUG: Log what API returned
    print("\n" + "="*60)
    print("[DEBUG] Open-Meteo Air Quality API Response")
    print("="*60)
    print(f"Available keys: {list(aqi_data.keys())}")
    print("Pollutant values:")
    print(f"  pm25 (pm2_5): {aqi_data.get('pm2_5', 'MISSING')}")
    print(f"  pm10: {aqi_data.get('pm10', 'MISSING')}")
    print(f"  no2 (nitrogen_dioxide): {aqi_data.get('nitrogen_dioxide', 'MISSING')}")
    print(f"  co (carbon_monoxide): {aqi_data.get('carbon_monoxide', 'MISSING')}")
    print(f"  o3 (ozone): {aqi_data.get('ozone', 'MISSING')}")
    print(f"  so2 (dust): {aqi_data.get('dust', 'MISSING')}")
    print(f"  aqi (european_aqi): {aqi_data.get('european_aqi', 'MISSING')}")
    print("="*60 + "\n")
    
    pollutants = {
        "aqi": to_float(aqi_data.get("european_aqi", np.nan)),
        "pm25": to_float(aqi_data.get("pm2_5", np.nan)),
        "pm10": to_float(aqi_data.get("pm10", np.nan)),
        "no2": to_float(aqi_data.get("nitrogen_dioxide", np.nan)),
        "co": to_float(aqi_data.get("carbon_monoxide", np.nan)),
        "o3": to_float(aqi_data.get("ozone", np.nan)),
        "so2": to_float(aqi_data.get("dust", np.nan)),
    }
    
    # CHECK: Are we getting real data or NULLs?
    null_count = sum(1 for v in pollutants.values() if np.isnan(v) or v is None)
    total_cols = len(pollutants)
    
    if null_count >= total_cols - 1:
        print("\n" + "!"*60)
        print("[CRITICAL] Open-Meteo returned NULL for most/all pollutants!")
        print("This will cause NaN cascade in predictions.")
        print("Possible causes:")
        print("  1. Coordinates (lat/lon) are incorrect")
        print("  2. Open-Meteo API is down or rate-limited")
        print("  3. Location has no air quality data available")
        print(f"  4. Check: LAT={LAT}, LON={LON}")
        print("!"*60 + "\n")
    
    return pollutants


def compute_features(pollutants: dict, weather: dict, timestamp: datetime) -> dict:
    """Compute all model features from raw data."""
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    day_of_year = timestamp.timetuple().tm_yday

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    wind_rad = np.radians(weather.get("wind_direction_10m", 0) or 0)
    wind_speed = weather.get("wind_speed_10m", 0) or 0
    wind_u = -wind_speed * np.sin(wind_rad)
    wind_v = -wind_speed * np.cos(wind_rad)

    is_rush_hour = int(hour in range(7, 10) or hour in range(17, 20))
    is_weekend = int(day_of_week >= 5)

    feature_row = {
        "timestamp": timestamp,
        "city": "islamabad_rawalpindi",
        **{k: float(v) if not np.isnan(v) else None for k, v in pollutants.items()},
        "temperature": weather.get("temperature_2m"),
        "humidity": weather.get("relative_humidity_2m"),
        "wind_speed": wind_speed,
        "wind_direction": weather.get("wind_direction_10m"),
        "pressure": weather.get("surface_pressure"),
        "precipitation": weather.get("precipitation"),
        "cloud_cover": weather.get("cloud_cover"),
        "wind_u": round(wind_u, 4),
        "wind_v": round(wind_v, 4),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "day_of_year": day_of_year,
        "hour_sin": round(hour_sin, 4),
        "hour_cos": round(hour_cos, 4),
        "month_sin": round(month_sin, 4),
        "month_cos": round(month_cos, 4),
        "dow_sin": round(dow_sin, 4),
        "dow_cos": round(dow_cos, 4),
        "is_rush_hour": is_rush_hour,
        "is_weekend": is_weekend,
    }
    return feature_row


def save_features_to_csv(feature_row: dict):
    """Save feature row to local CSV file (append mode)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame([feature_row])
    
    # If CSV exists, append; otherwise create new
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, mode="w", header=True, index=False)
    print(f"[OK] Features saved to CSV: {CSV_PATH}")


def store_features(feature_row: dict):
    """Connect to Hopsworks and upsert the feature row."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=CONFIG["feature_store"]["project_name"],
    )
    fs = project.get_feature_store()

    # Get the existing feature group instead of creating new versions
    try:
        fg = fs.get_feature_group(
            CONFIG["feature_store"]["feature_group_name"],
            version=CONFIG["feature_store"]["feature_group_version"]
        )
        print(f"[OK] Connected to existing feature group: {CONFIG['feature_store']['feature_group_name']} v{CONFIG['feature_store']['feature_group_version']}")
    except Exception as e:
        print(f"[WARN] Could not find existing feature group: {e}")
        print("Creating new feature group...")
        fg = fs.get_or_create_feature_group(
            name=CONFIG["feature_store"]["feature_group_name"],
            version=CONFIG["feature_store"]["feature_group_version"],
            description="Hourly AQI and weather features for Islamabad/Rawalpindi",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=True,
            time_travel_format="HUDI",
        )

    df = pd.DataFrame([feature_row])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric_cols = ["aqi", "pm25", "pm10", "no2", "co", "o3", "so2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    int_cols = [
        "humidity",
        "wind_direction",
        "cloud_cover",
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "is_rush_hour",
        "is_weekend",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(0)
        df[col] = df[col].apply(lambda v: int(v) if pd.notna(v) else None)
    fg.insert(
    df,
    write_options={
        "wait_for_job": True,
        "start_offline_materialization": True,
    },
)
    print(f"[OK] Features stored to Hopsworks at {feature_row['timestamp']}")


def run_feature_pipeline():
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    try:
        print("Fetching air quality data from Open-Meteo...")
        aqi_raw = fetch_aqi_data(LAT, LON)
        pollutants = extract_pollutants(aqi_raw)

        print("Fetching weather data...")
        weather = fetch_weather_data(LAT, LON)

        print("Computing features...")
        features = compute_features(pollutants, weather, timestamp)

        # Try to store in Hopsworks
        print("Storing features in Hopsworks...")
        try:
            store_features(features)
            print("[SUCCESS] Features written to Hopsworks ✓")
        except Exception as e:
            print(f"[ERROR] Hopsworks write failed: {type(e).__name__}: {e}")
            raise
        
        print("Saving features to local CSV...")
        save_features_to_csv(features)
        
        print("=" * 50)
        print(f"[COMPLETE] Feature pipeline run at {timestamp}")
        print("=" * 50)

        return features

    except Exception as e:
        print("=" * 50)
        print(f"[CRITICAL ERROR] Pipeline failed: {type(e).__name__}")
        print(f"Message: {e}")
        print("=" * 50)
        raise


if __name__ == "__main__":
    run_feature_pipeline()