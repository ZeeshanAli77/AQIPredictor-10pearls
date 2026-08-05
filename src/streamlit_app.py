"""
Islamabad Air Quality Forecasting System
Real-time AQI predictions for Islamabad, Pakistan
Developed by Zeeshan
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import hopsworks
from dotenv import load_dotenv
import yaml

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
CITY = CONFIG.get("city", {})
LAT = CITY.get("latitude")
LON = CITY.get("longitude")

st.set_page_config(
    page_title="Islamabad AQI Predictor - Zeeshan",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for branding
st.markdown("""
    <style>
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .header-banner p {
        margin: 5px 0 0 0;
        font-size: 1.1em;
        opacity: 0.9;
    }
    .developer-tag {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        margin-top: 10px;
    }
    .metric-card {
        background: #f0f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
    }
    .forecast-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #e8f4f8;
        color: #0d3b4f;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0288d1;
        margin: 10px 0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

AQI_LEVELS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory. Perfect for outdoor activities."),
    (51, 100, "Moderate", "#b8860b", "Sensitive individuals may feel mild discomfort during prolonged outdoor exposure."),
    (101, 150, "Unhealthy (Sensitive)", "#ff7e00", "Sensitive groups should limit exposure. Consider using masks."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may experience health effects. Masks recommended."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: serious effects for everyone. Avoid outdoor activity."),
    (301, 500, "Hazardous", "#7e0023", "Emergency conditions. Avoid all outdoor activity immediately."),
]


def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.getenv(name, "")


def classify_aqi(value: float) -> tuple:
    try:
        numeric = round(float(value))
    except (TypeError, ValueError):
        numeric = 0
    for lo, hi, label, color, advice in AQI_LEVELS:
        if lo <= numeric <= hi:
            return label, color, advice
    return "Hazardous", "#7e0023", "Emergency conditions."


def get_aqi_gauge(value: float) -> go.Figure:
    label, color, _ = classify_aqi(value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<span style='color:#ffffff;'>Current AQI</span><br><span style='font-size:14px;color:{color}'>{label}</span>"},
            number={"valueformat": ".0f", "font": {"color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#666666"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#00e400"},
                    {"range": [51, 100], "color": "#b8860b"},
                    {"range": [101, 150], "color": "#ff7e00"},
                    {"range": [151, 200], "color": "#ff0000"},
                    {"range": [201, 300], "color": "#8f3f97"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320, 
        margin=dict(t=60, b=20), 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"}
    )
    return fig


@st.cache_data(ttl=3600)
def load_current_aqi():
    token = get_secret("AQICN_TOKEN")
    if not token:
        return {
            "aqi": 0,
            "pm25": 0,
            "pm10": 0,
            "no2": 0,
            "temperature": 0,
            "humidity": 0,
            "wind": 0,
            "updated": "Missing AQICN token",
        }
    url = f"https://api.waqi.info/feed/geo:{LAT};{LON}/?token={token}"
    resp = requests.get(url, timeout=30)
    data = resp.json().get("data", {})
    aqi_value = data.get("aqi", 0)
    try:
        aqi_value = float(aqi_value)
    except (TypeError, ValueError):
        aqi_value = 0.0
    iaqi = data.get("iaqi", {})
    return {
        "aqi": aqi_value,
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "no2": iaqi.get("no2", {}).get("v", 0),
        "temperature": iaqi.get("t", {}).get("v", 0),
        "humidity": iaqi.get("h", {}).get("v", 0),
        "wind": iaqi.get("w", {}).get("v", 0),
        "updated": data.get("time", {}).get("s", ""),
    }


@st.cache_data(ttl=3600)
def load_forecast():
    try:
        project = hopsworks.login(
            api_key_value=get_secret("HOPSWORKS_API_KEY")
        )
        mr = project.get_model_registry()
        
        # Load 24h model
        model_24 = mr.get_best_model("aqi_predictor_target_aqi_24h", metric="rmse", direction="min")
        saved_model_dir_24 = model_24.download()
        clf_24 = joblib.load(os.path.join(saved_model_dir_24, "aqi_target_aqi_24h_model.pkl"))

        # Load 48h model
        model_48 = mr.get_best_model("aqi_predictor_target_aqi_48h", metric="rmse", direction="min")
        saved_model_dir_48 = model_48.download()
        clf_48 = joblib.load(os.path.join(saved_model_dir_48, "aqi_target_aqi_48h_model.pkl"))

        # Load 72h model
        model_72 = mr.get_best_model("aqi_predictor_target_aqi_72h", metric="rmse", direction="min")
        saved_model_dir_72 = model_72.download()
        clf_72 = joblib.load(os.path.join(saved_model_dir_72, "aqi_target_aqi_72h_model.pkl"))

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

        feature_cols = [
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

        X = latest[feature_cols]
        if X.isna().any(axis=None):
            valid = df[feature_cols].dropna()
            if valid.empty:
                raise RuntimeError("Latest features contain NaNs; check history.")
            X = valid.tail(1)
            
        pred_24 = float(clf_24.predict(X)[0])
        pred_48 = float(clf_48.predict(X)[0])
        pred_72 = float(clf_72.predict(X)[0])

        today = datetime.now()
        return [
            {"date": (today + timedelta(days=1)).strftime("%A, %b %d"), "aqi": max(0, pred_24)},
            {"date": (today + timedelta(days=2)).strftime("%A, %b %d"), "aqi": max(0, pred_48)},
            {"date": (today + timedelta(days=3)).strftime("%A, %b %d"), "aqi": max(0, pred_72)},
        ]
    except Exception as exc:
        st.warning(f"Could not load live model predictions. Showing sample data. ({exc})")
        today = datetime.now()
        return [
            {"date": (today + timedelta(days=1)).strftime("%A, %b %d"), "aqi": 145},
            {"date": (today + timedelta(days=2)).strftime("%A, %b %d"), "aqi": 130},
            {"date": (today + timedelta(days=3)).strftime("%A, %b %d"), "aqi": 110},
        ]


def main():
    # Custom header banner
    st.markdown("""
        <div class="header-banner">
            <h1>💨 Islamabad AQI Predictor</h1>
            <p>Real-time Air Quality Forecasting for Islamabad, Pakistan</p>
            <div class="developer-tag">🔬 Developed by Zeeshan</div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading real-time AQI data for Islamabad..."):
        current = load_current_aqi()
        forecast = load_forecast()

    label, color, advice = classify_aqi(current["aqi"])

    # Current AQI Section
    st.subheader("📊 Current Air Quality Status")
    
    col_gauge, col_info = st.columns([1, 1.2])
    
    with col_gauge:
        st.plotly_chart(get_aqi_gauge(current["aqi"]), use_container_width=True)
    
    with col_info:
        st.markdown(f"### {label}")
        st.markdown(f"<div class='info-box'>{advice}</div>", unsafe_allow_html=True)
        
        if current.get('updated'):
            st.caption(f"📅 Last updated: {current.get('updated')}")

    # Pollutants & Weather Metrics
    st.subheader("🌡️ Environmental Metrics")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("PM2.5", f"{current['pm25']:.1f} µg/m³", delta="Fine Particulates")
    with m2:
        st.metric("PM10", f"{current['pm10']:.1f} µg/m³", delta="Coarse Particulates")
    with m3:
        st.metric("NO₂", f"{current['no2']:.1f} ppb", delta="Nitrogen Dioxide")
    with m4:
        st.metric("Temperature", f"{current['temperature']:.1f}°C", delta="Current Temp")

    st.divider()

    # 3-Day Forecast Section
    st.subheader("📈 72-Hour AQI Forecast")
    st.markdown("*Predictions for the next 3 days based on ML models trained on Islamabad data*")
    
    cols = st.columns(3, gap="medium")
    for i, day in enumerate(forecast):
        lbl, clr, day_advice = classify_aqi(day["aqi"])
        with cols[i]:
            st.markdown(f"""
                <div class="forecast-card" style="background: linear-gradient(135deg, {clr}20, {clr}40); border: 2px solid {clr};">
                    <h4 style="margin: 0; color: #f5f5f5;">{day['date']}</h4>
                    <div style="font-size: 48px; font-weight: bold; color: {clr}; margin: 15px 0;">
                        {day['aqi']:.0f}
                    </div>
                    <p style="margin: 0; color: {clr}; font-weight: 600;">{lbl}</p>
                    <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #d8d8d8;">{day_advice[:50]}...</p>
                </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Chart Section
    st.subheader("📉 Forecast Trend Analysis")
    forecast_df = pd.DataFrame(forecast)
    fig = px.bar(
        forecast_df,
        x="date",
        y="aqi",
        color="aqi",
        color_continuous_scale=["#00e400", "#b8860b", "#ff7e00", "#ff0000", "#8f3f97"],
        range_color=[0, 300],
        labels={"aqi": "Predicted AQI", "date": "Date"},
        title="Next 3 Days - AQI Predictions with Health Thresholds",
        height=400
    )
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text="Moderate Threshold (100)",
        annotation_position="right"
    )
    fig.add_hline(
        y=150,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Unhealthy Threshold (150)",
        annotation_position="right"
    )
    fig.update_layout(
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240,244,248,0.5)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Feature Importance Section
    if os.path.exists("artifacts/shap_summary.png"):
        st.subheader("🔍 Model Feature Importance (SHAP Analysis)")
        st.markdown("*These features have the strongest influence on our AQI predictions*")
        col1, col2, col3 = st.columns([0.5, 3, 0.5])
        with col2:
            st.image("artifacts/shap_summary.png", caption="SHAP Summary Plot - Feature Impact on Predictions", use_container_width=True)

    st.divider()

    # Health Alerts
    st.subheader("⚠️ Health Alerts & Recommendations")
    
    if any(d["aqi"] > 150 for d in forecast):
        st.error(
            "🚨 **ALERT**: Unhealthy air quality forecasted in the next 3 days. "
            "Avoid outdoor activities, use N95 masks, and keep windows closed."
        )
    elif any(d["aqi"] > 100 for d in forecast):
        st.warning(
            "⚠️ **CAUTION**: Moderate to unhealthy air quality expected. "
            "Sensitive groups (children, elderly, asthmatics) should limit outdoor exposure."
        )
    else:
        st.success(
            "✅ **GOOD NEWS**: Air quality is expected to remain good. "
            "Outdoor activities should be safe for most people."
        )

    st.divider()

    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee;'>
            <p><strong>Islamabad AQI Predictor</strong> | Developed by Zeeshan</p>
            <p style='font-size: 0.85em;'>
                Data Sources: AQICN (Ground Sensors) | Open-Meteo (Weather & Climate) | 
                Hopsworks (Feature Store & Model Registry)<br>
                ML Models: Random Forest & Ridge Regression | Explainability: SHAP
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()