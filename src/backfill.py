"""
Backfill Script: Generate historical features for training data.
Run once before training the first model.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

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


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather from Open-Meteo Historical API (free)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        "wind_direction_10m,surface_pressure,precipitation,cloud_cover"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_historical_aqi_openmeteo(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical air quality data from Open-Meteo Air Quality API.
    Provides pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, ozone, dust, european_aqi.
    """
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,dust,european_aqi"
        f"&start_date={start_date}&end_date={end_date}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={
            "pm2_5": "pm25",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "ozone": "o3",
            "european_aqi": "aqi",
        },
        inplace=True,
    )
    return df


def build_historical_feature_df(
    weather_df: pd.DataFrame,
    aqi_df: pd.DataFrame,
    city: str = "islamabad_rawalpindi",
) -> pd.DataFrame:
    """Merge weather and AQI data, compute all features."""
    merged = pd.merge(weather_df, aqi_df, on="time", how="inner")

    rows = []
    for _, row in merged.iterrows():
        ts = row["time"].to_pydatetime()
        hour = ts.hour
        dow = ts.weekday()
        month = ts.month
        doy = ts.timetuple().tm_yday

        wind_rad = np.radians(row.get("wind_direction_10m", 0) or 0)
        wind_speed = row.get("wind_speed_10m", 0) or 0
        wind_u = -wind_speed * np.sin(wind_rad)
        wind_v = -wind_speed * np.cos(wind_rad)

        features = {
            "timestamp": ts,
            "city": city,
            "aqi": float(row.get("aqi", np.nan)),
            "pm25": float(row.get("pm25", np.nan)),
            "pm10": float(row.get("pm10", np.nan)),
            "no2": float(row.get("no2", np.nan)),
            "co": float(row.get("co", np.nan)),
            "o3": float(row.get("o3", np.nan)),
            "so2": float(row.get("dust", np.nan)),
            "temperature": float(row.get("temperature_2m", np.nan)),
            "humidity": float(row.get("relative_humidity_2m", np.nan)),
            "wind_speed": float(wind_speed),
            "wind_direction": float(row.get("wind_direction_10m", np.nan)),
            "pressure": float(row.get("surface_pressure", np.nan)),
            "precipitation": float(row.get("precipitation", np.nan)),
            "cloud_cover": float(row.get("cloud_cover", np.nan)),
            "wind_u": round(wind_u, 4),
            "wind_v": round(wind_v, 4),
            "hour": hour,
            "day_of_week": dow,
            "month": month,
            "day_of_year": doy,
            "hour_sin": round(np.sin(2 * np.pi * hour / 24), 4),
            "hour_cos": round(np.cos(2 * np.pi * hour / 24), 4),
            "month_sin": round(np.sin(2 * np.pi * month / 12), 4),
            "month_cos": round(np.cos(2 * np.pi * month / 12), 4),
            "dow_sin": round(np.sin(2 * np.pi * dow / 7), 4),
            "dow_cos": round(np.cos(2 * np.pi * dow / 7), 4),
            "is_rush_hour": int(hour in range(7, 10) or hour in range(17, 20)),
            "is_weekend": int(dow >= 5),
        }
        rows.append(features)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
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

    df = df.dropna(subset=["aqi", "target_aqi_24h"])
    return df


def run_backfill(months_back: int = 6) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=months_back * 30)

    print(f"Backfilling from {start_date} to {end_date}...")

    all_weather = []
    all_aqi = []

    current = start_date
    while current < end_date:
        chunk_end = min(current + timedelta(days=30), end_date)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        print(f"  Fetching {start_str} -> {end_str}...")
        weather_chunk = fetch_historical_weather(LAT, LON, start_str, end_str)
        aqi_chunk = fetch_historical_aqi_openmeteo(LAT, LON, start_str, end_str)

        all_weather.append(weather_chunk)
        all_aqi.append(aqi_chunk)

        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    weather_df = pd.concat(all_weather, ignore_index=True)
    aqi_df = pd.concat(all_aqi, ignore_index=True)

    print(f"  Total records: {len(weather_df)} weather, {len(aqi_df)} AQI")

    features_df = build_historical_feature_df(weather_df, aqi_df)
    print(
        f"  Features computed: {len(features_df)} rows, {len(features_df.columns)} columns"
    )

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
        description="Hourly AQI features - Islamabad/Rawalpindi",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )

    base_cols = [
        "timestamp",
        "city",
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "co",
        "o3",
        "so2",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
        "precipitation",
        "cloud_cover",
        "wind_u",
        "wind_v",
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_rush_hour",
        "is_weekend",
    ]

    insert_df = features_df[base_cols].copy()
    insert_df["timestamp"] = pd.to_datetime(
        insert_df["timestamp"], utc=True, errors="coerce"
    )

    float_cols = [
        "aqi",
        "pm25",
        "pm10",
        "no2",
        "co",
        "o3",
        "so2",
        "temperature",
        "wind_speed",
        "pressure",
        "precipitation",
        "wind_u",
        "wind_v",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    ]
    for col in float_cols:
        insert_df[col] = pd.to_numeric(insert_df[col], errors="coerce")

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
        insert_df[col] = pd.to_numeric(insert_df[col], errors="coerce").round(0)
        insert_df[col] = insert_df[col].apply(
            lambda value: int(value) if pd.notna(value) else None
        )

    fg.insert(insert_df, write_options={"wait_for_job": True})
    print(f"[DONE] Backfill complete. {len(insert_df)} rows stored.")

    return features_df


if __name__ == "__main__":
    df = run_backfill(months_back=6)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    df.to_csv(os.path.join(BASE_DIR, "data", "backfill_preview.csv"), index=False)
    print(df[["timestamp", "aqi", "pm25", "temperature"]].tail(10))