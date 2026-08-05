"""
Feature Pipeline: Fetch, compute, and store AQI features for Islamabad/Rawalpindi.
Runs every hour via GitHub Actions.
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


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
CITY = CONFIG["city"]
LAT, LON = CITY["latitude"], CITY["longitude"]

AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")


def fetch_aqicn_data(station: str) -> dict:
    """Fetch real-time AQI data from AQICN API for a given station."""
    if not AQICN_TOKEN:
        raise RuntimeError("AQICN_TOKEN is missing. Set it in your environment.")
    url = f"https://api.waqi.info/feed/{station}/?token={AQICN_TOKEN}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "ok":
        raise ValueError(f"AQICN API error: {data.get('data', 'Unknown error')}")
    return data["data"]


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


def extract_pollutants(aqicn_data: dict) -> dict:
    """Extract pollutant readings from AQICN response."""
    def to_float(value) -> float:
        if value in (None, "-", ""):
            return np.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    iaqi = aqicn_data.get("iaqi", {})
    return {
        "aqi": to_float(aqicn_data.get("aqi", np.nan)),
        "pm25": to_float(iaqi.get("pm25", {}).get("v", np.nan)),
        "pm10": to_float(iaqi.get("pm10", {}).get("v", np.nan)),
        "no2": to_float(iaqi.get("no2", {}).get("v", np.nan)),
        "co": to_float(iaqi.get("co", {}).get("v", np.nan)),
        "o3": to_float(iaqi.get("o3", {}).get("v", np.nan)),
        "so2": to_float(iaqi.get("so2", {}).get("v", np.nan)),
    }


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


def store_features(feature_row: dict):
    """Connect to Hopsworks and upsert the feature row."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=CONFIG["feature_store"]["project_name"],
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=CONFIG["feature_store"]["feature_group_name"],
        version=CONFIG["feature_store"]["feature_group_version"],
        description="Hourly AQI and weather features for Islamabad/Rawalpindi",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
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
            "wait_for_job": False,
            "start_offline_materialization": False,
        },
    )
    print(f"[OK] Features stored at {feature_row['timestamp']}")


def run_feature_pipeline():
    timestamp = datetime.now(timezone.utc)

    print("Fetching AQICN data...")
    aqicn_raw = fetch_aqicn_data(CITY["aqicn_station"])
    pollutants = extract_pollutants(aqicn_raw)

    print("Fetching weather data...")
    weather = fetch_weather_data(LAT, LON)

    print("Computing features...")
    features = compute_features(pollutants, weather, timestamp)

    print("Storing features in Hopsworks...")
    store_features(features)

    return features


if __name__ == "__main__":
    run_feature_pipeline()
