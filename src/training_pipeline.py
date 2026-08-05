"""
Training Pipeline: Train, evaluate, and register AQI prediction models.
Runs daily via GitHub Actions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hopsworks
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOCAL_BACKFILL_PATH = os.path.join(BASE_DIR, "data", "backfill_preview.csv")

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

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
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


def fetch_training_data() -> pd.DataFrame:
    """Pull features from Hopsworks Feature Store or fallback to local CSV."""
    if not HOPSWORKS_API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing. Set it in your environment.")

    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()

        fg = fs.get_feature_group("aqi_features", version=1)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")
        return df
    except Exception as exc:
        if os.path.exists(LOCAL_BACKFILL_PATH):
            print(
                "[WARN] Hopsworks read failed; using local backfill CSV: "
                f"{LOCAL_BACKFILL_PATH}"
            )
            df = pd.read_csv(LOCAL_BACKFILL_PATH)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("timestamp")
            return df
        raise RuntimeError(
            "Hopsworks read failed and local backfill CSV not found. "
            "Run backfill first."
        ) from exc


def split_time_series(df: pd.DataFrame, val_fraction: float = 0.15):
    """Split preserving chronological order to prevent leakage."""
    split_idx = int(len(df) * (1 - val_fraction))
    train = df.iloc[:split_idx]
    val = df.iloc[split_idx:]
    return train, val


def evaluate_model(model, X_val, y_val) -> dict:
    preds = model.predict(X_val)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_val, preds))), 4),
        "mae": round(float(mean_absolute_error(y_val, preds)), 4),
        "r2": round(float(r2_score(y_val, preds)), 4),
    }


def train_models(X_train, y_train, X_val, y_val, target_name: str):
    models = {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"  Training {name} for {target_name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        results[name] = {"model": model, "metrics": metrics}
        print(f"    RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']}")

    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    print(f"  Best model for {target_name}: {best_name}")
    return results[best_name]["model"], results[best_name]["metrics"], best_name


def compute_shap_values(model, X_val: pd.DataFrame, model_name: str):
    """Compute and save SHAP feature importance plot."""
    print("  Computing SHAP values...")
    try:
        if model_name in ("xgboost", "random_forest"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
        else:
            scaled = model.named_steps["scaler"].transform(X_val)
            explainer = shap.LinearExplainer(model.named_steps["model"], scaled)
            shap_values = explainer.shap_values(scaled)

        shap.summary_plot(shap_values, X_val, show=False, max_display=15)
        plt.tight_layout()
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/shap_summary.png", dpi=150)
        plt.close()
        print("  SHAP plot saved to artifacts/shap_summary.png")
    except Exception as exc:
        print(f"  SHAP warning: {exc}")


def register_model(project, model, model_name: str, metrics: dict, target: str, feature_cols: list):
    """Save model artifacts and register in Hopsworks Model Registry."""
    os.makedirs("model_artifacts", exist_ok=True)

    model_path = f"model_artifacts/aqi_{target}_model.pkl"
    joblib.dump(model, model_path)

    meta = {
        "model_name": model_name,
        "target": target,
        "features": feature_cols,
        "metrics": metrics,
        "trained_at": datetime.utcnow().isoformat(),
        "city": "islamabad_rawalpindi",
    }
    with open("model_artifacts/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    mr = project.get_model_registry()
    hw_model = mr.sklearn.create_model(
        name=f"aqi_predictor_{target}",
        metrics=metrics,
        description=f"AQI {target} prediction | {model_name} | Islamabad/Rawalpindi",
        input_example=pd.DataFrame([{col: 0.0 for col in feature_cols}]),
        feature_view=None,
    )
    hw_model.save("model_artifacts/")
    print(f"  Model registered: aqi_predictor_{target} v{hw_model.version}")
    return hw_model


def run_training_pipeline():
    print("Fetching training data from Feature Store...")
    df = fetch_training_data()
    df = add_time_series_features(df)
    print(f"  Total samples: {len(df)}")

    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    train_df, val_df = split_time_series(df)

    X_train = train_df[FEATURE_COLS]
    X_val = val_df[FEATURE_COLS]

    all_metrics = {}

    for target in TARGET_COLS:
        print("=" * 50)
        print(f"Training for target: {target}")
        y_train = train_df[target]
        y_val = val_df[target]

        best_model, metrics, best_name = train_models(
            X_train, y_train, X_val, y_val, target
        )
        all_metrics[target] = {**metrics, "model_type": best_name}

        if target == "target_aqi_24h":
            compute_shap_values(best_model, X_val, best_name)

        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        register_model(project, best_model, best_name, metrics, target, FEATURE_COLS)

    print("=== TRAINING SUMMARY ===")
    for target, metrics in all_metrics.items():
        print(
            f"{target}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R2={metrics['r2']} ({metrics['model_type']})"
        )

    return all_metrics


if __name__ == "__main__":
    run_training_pipeline()
