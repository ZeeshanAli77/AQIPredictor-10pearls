"""
Islamabad Air Quality Forecasting System - Enhanced UI
Real-time AQI predictions for Islamabad, Pakistan
Developed by Zeeshan
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

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

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main background and general styling */
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 0;
    }
    
    /* Header banner - more premium look */
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 50px 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .header-banner h1 {
        margin: 0;
        font-size: 3em;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .header-banner p {
        margin: 15px 0 0 0;
        font-size: 1.2em;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .developer-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.95em;
        margin-top: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Current AQI card - prominent display */
    .aqi-current-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .aqi-current-card h2 {
        margin: 20px 0;
        color: #ffffff;
        font-size: 2.5em;
    }
    
    .aqi-value {
        font-size: 5em;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Forecast cards - sleek design */
    .forecast-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        color: #ffffff;
    }
    
    .forecast-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .forecast-card h4 {
        margin: 0 0 15px 0;
        font-size: 1.1em;
        opacity: 0.9;
        font-weight: 600;
    }
    
    .forecast-aqi {
        font-size: 3.5em;
        font-weight: 900;
        margin: 20px 0;
    }
    
    .forecast-label {
        font-size: 1.2em;
        font-weight: 700;
        margin: 15px 0;
    }
    
    /* Metrics grid */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
        border: 2px solid rgba(255, 255, 255, 0.15);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: 800;
        color: #667eea;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info/Alert boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
    
    .alert-box-good {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1));
        border-left-color: #22c55e;
    }
    
    .alert-box-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        border-left-color: #f59e0b;
    }
    
    .alert-box-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        border-left-color: #ef4444;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 40px 0;
    }
    
    /* Section headers */
    h2 {
        color: #ffffff;
        font-size: 2em;
        margin-top: 40px !important;
        margin-bottom: 20px !important;
    }
    
    h3 {
        color: #ffffff;
        margin-bottom: 15px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #888;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 40px;
        font-size: 0.9em;
    }
    
    /* Overall text color */
    .stMarkdown, p, span {
        color: #ffffff;
    }
    
    /* Plotly charts background */
    .plotly-container {
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

AQI_LEVELS = [
    (0, 50, "Good", "#22c55e", "Air quality is satisfactory. Perfect for outdoor activities."),
    (51, 100, "Moderate", "#f59e0b", "Sensitive individuals may feel mild discomfort during prolonged outdoor exposure."),
    (101, 150, "Unhealthy (Sensitive)", "#ff6b35", "Sensitive groups should limit exposure. Consider using masks."),
    (151, 200, "Unhealthy", "#ef4444", "Everyone may experience health effects. Masks recommended."),
    (201, 300, "Very Unhealthy", "#a855f7", "Health alert: serious effects for everyone. Avoid outdoor activity."),
    (301, 500, "Hazardous", "#7e0023", "Emergency conditions. Avoid all outdoor activity immediately."),
]


def get_secret(name: str) -> str:
    try:
        value = str(st.secrets[name])
        print(f"✓ Got {name} from st.secrets: {value[:20] if len(value) > 20 else value}...")
        return value
    except Exception as e:
        value = os.getenv(name, "")
        print(f"✗ st.secrets failed for {name}, got from env: {value[:20] if value and len(value) > 20 else value}... (error: {e})")
        return value


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
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"Current AQI<br><span style='font-size:14px;'>{label}</span>"},
            number={"valueformat": ".0f", "font": {"color": "#ffffff", "size": 40}},
            gauge={
                "axis": {"range": [0, 300], "tickwidth": 1, "tickcolor": "#555555"},
                "bar": {"color": color, "thickness": 0.2},
                "steps": [
                    {"range": [0, 50], "color": "rgba(34, 197, 94, 0.3)"},
                    {"range": [51, 100], "color": "rgba(245, 158, 11, 0.3)"},
                    {"range": [101, 150], "color": "rgba(255, 107, 53, 0.3)"},
                    {"range": [151, 200], "color": "rgba(239, 68, 68, 0.3)"},
                    {"range": [201, 300], "color": "rgba(168, 85, 247, 0.3)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff", "size": 12},
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

def get_hopsworks_project():
    """Cache the Hopsworks project connection"""
    return hopsworks.login(
        api_key_value=get_secret("HOPSWORKS_API_KEY"),
        host=get_secret("HOPSWORKS_HOST")
    )

def get_hopsworks_models(project):
    """Load the latest trained models from Hopsworks Model Registry
    
    Automatically fetches the newest version of each model.
    As new models are trained (v21, v22, v30, etc.), they're used without code changes.
    """

    mr = project.get_model_registry()

    def load_latest_model(model_name: str, pkl_filename: str) -> tuple:
        """Get the latest version of a model and load it"""
        model = mr.get_model(model_name)
        
        # Get all available versions
        all_versions = model.get_all_versions()
        
        if not all_versions:
            raise RuntimeError(f"No versions found for model: {model_name}")
        
        # Get the latest version (highest version number)
        latest_version = max(all_versions, key=lambda v: v.version)
        version_num = latest_version.version
        
        print(f"✓ Loading {model_name} version {version_num}")
        
        # Download the latest model
        saved_model_dir = latest_version.download()
        
        # Load the model
        clf = joblib.load(
            os.path.join(saved_model_dir, pkl_filename)
        )
        
        return clf, version_num

    # Load all three models with their latest versions
    clf_24, v24 = load_latest_model(
        "aqi_predictor_target_aqi_24h",
        "aqi_target_aqi_24h_model.pkl"
    )
    
    clf_48, v48 = load_latest_model(
        "aqi_predictor_target_aqi_48h",
        "aqi_target_aqi_48h_model.pkl"
    )
    
    clf_72, v72 = load_latest_model(
        "aqi_predictor_target_aqi_72h",
        "aqi_target_aqi_72h_model.pkl"
    )
    
    print(f"\n{'='*70}")
    print(f"✓ LOADED LATEST MODELS FROM HOPSWORKS")
    print(f"  📦 24h forecast model: v{v24}")
    print(f"  📦 48h forecast model: v{v48}")
    print(f"  📦 72h forecast model: v{v72}")
    print(f"{'='*70}\n")

    return clf_24, clf_48, clf_72


def load_forecast():
    """Load 3-day AQI forecast from Hopsworks models"""
    try:
        project = get_hopsworks_project()
        clf_24, clf_48, clf_72 = get_hopsworks_models(project)

        fs = project.get_feature_store()
        fg = fs.get_feature_group("aqi_features", version=7)
        df = fg.read()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
            errors="coerce"
        )

        df = df.sort_values("timestamp")

        if len(df) < 25:
            raise RuntimeError(
                "Not enough history to compute lag features."
            )

        df["aqi"] = pd.to_numeric(
            df["aqi"],
            errors="coerce"
        )

        # Create lag features
        df["aqi_lag_1h"] = df["aqi"].shift(1)
        df["aqi_lag_3h"] = df["aqi"].shift(3)
        df["aqi_lag_6h"] = df["aqi"].shift(6)
        df["aqi_lag_24h"] = df["aqi"].shift(24)

        # Create derived features
        df["aqi_change_1h"] = (
            df["aqi"] - df["aqi_lag_1h"]
        )

        df["aqi_roll_3h"] = (
            df["aqi"].rolling(3).mean()
        )

        df["aqi_roll_6h"] = (
            df["aqi"].rolling(6).mean()
        )

        df["aqi_roll_24h"] = (
            df["aqi"].rolling(24).mean()
        )

        df["aqi_roll_std"] = (
            df["aqi"].rolling(6).std()
        )

        # Get latest available data
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

        # Handle missing values - use recent rolling stats instead of fallback
        if X.isna().any(axis=None):
            print("\n[WARN] Latest row has NaN features. Using recent data backfill...")
            recent = df[feature_cols].tail(48).fillna(method='ffill').fillna(method='bfill')
            if recent.empty or recent.isna().any(axis=None).any():
                raise RuntimeError("Cannot fill NaN features even with recent 48-hour history.")
            X = recent.tail(1)
            print("[OK] Features backfilled from recent data")

        # DEBUG: Print actual features being used
        print("\n" + "="*70)
        print("[DEBUG] FEATURES BEING FED TO MODELS:")
        print("="*70)
        print(X.to_string())
        print(f"\nNaN count per column:")
        print(X.isna().sum())
        print(f"\nLatest actual AQI in history: {df['aqi'].iloc[-1]:.1f}")
        data_age_minutes = (datetime.now(timezone.utc) - df['timestamp'].iloc[-1]).total_seconds() / 60
        print(f"Data age: {data_age_minutes:.1f} minutes")
        print("="*70 + "\n")

        # Make predictions
        pred_24 = float(
            clf_24.predict(X)[0]
        )

        pred_48 = float(
            clf_48.predict(X)[0]
        )

        pred_72 = float(
            clf_72.predict(X)[0]
        )

        # Debug information
        print("===== AQI FORECAST DEBUG =====")
        print(
            "Latest timestamp:",
            latest["timestamp"].iloc[0]
        )
        print(
            "Latest actual AQI:",
            latest["aqi"].iloc[0]
        )
        print(
            "24h prediction:",
            pred_24
        )
        print(
            "48h prediction:",
            pred_48
        )
        print(
            "72h prediction:",
            pred_72
        )
        print("==============================")

        # Generate forecast dates
        today = datetime.now()

        return [
            {
                "date": (
                    today + timedelta(days=1)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=1)
                ).strftime("%A, %B %d"),
                "aqi": max(0, pred_24),
            },
            {
                "date": (
                    today + timedelta(days=2)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=2)
                ).strftime("%A, %B %d"),
                "aqi": max(0, pred_48),
            },
            {
                "date": (
                    today + timedelta(days=3)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=3)
                ).strftime("%A, %B %d"),
                "aqi": max(0, pred_72),
            },
        ]

    except Exception as exc:
        st.warning(
            f"Could not load live model predictions. ({exc})"
        )

        today = datetime.now()

        return [
            {
                "date": (
                    today + timedelta(days=1)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=1)
                ).strftime("%A, %B %d"),
                "aqi": 145,
            },
            {
                "date": (
                    today + timedelta(days=2)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=2)
                ).strftime("%A, %B %d"),
                "aqi": 130,
            },
            {
                "date": (
                    today + timedelta(days=3)
                ).strftime("%a, %b %d"),
                "full_date": (
                    today + timedelta(days=3)
                ).strftime("%A, %B %d"),
                "aqi": 110,
            },
        ]


def main():
    # Clear cache button
    if st.sidebar.button("🔄 Clear Cache & Reload Models"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # Header
    st.markdown("""
        <div class="header-banner">
            <h1>💨 Islamabad AQI Predictor</h1>
            <p>Real-time Air Quality Forecasting & 72-Hour Predictions</p>
            <div class="developer-tag">🔬 Developed by Zeeshan</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Loading real-time data..."):
        current = load_current_aqi()
        forecast = load_forecast()

    label, color, advice = classify_aqi(current["aqi"])

    # Current AQI - Prominent Display
    st.markdown("### 📊 Current Air Quality")
    col_gauge, col_info = st.columns([1.2, 1])
    
    with col_gauge:
        st.plotly_chart(get_aqi_gauge(current["aqi"]), use_container_width=True)
    
    with col_info:
        st.markdown(f"""
            <div class="aqi-current-card">
                <h2>{label}</h2>
                <div class="aqi-value" style="color: {color};">{current['aqi']:.0f}</div>
                <div class="info-box" style="margin-top: 20px; border-left-color: {color};">
                    {advice}
                </div>
                <p style="margin-top: 20px; opacity: 0.7; font-size: 0.9em;">
                    📅 {current.get('updated', 'Recently updated')}
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Environmental Metrics
    st.markdown("### 🌡️ Air & Weather Conditions")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    metrics = [
        (m1, "PM2.5", f"{current['pm25']:.1f}", "µg/m³"),
        (m2, "PM10", f"{current['pm10']:.1f}", "µg/m³"),
        (m3, "NO₂", f"{current['no2']:.1f}", "ppb"),
        (m4, "🌡️ Temp", f"{current['temperature']:.1f}", "°C"),
        (m5, "💧 RH", f"{current['humidity']:.0f}", "%"),
        (m6, "💨 Wind", f"{current['wind']:.1f}", "m/s"),
    ]
    
    for col, label, value, unit in metrics:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size: 0.85em; opacity: 0.7;">{unit}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 72-Hour Forecast
    st.markdown("### 📈 3-Day AQI Forecast")
    st.markdown("*Predictions automatically update daily with new data*")
    
    cols = st.columns(3, gap="large")
    for i, day in enumerate(cols):
        forecast_data = forecast[i]
        lbl, clr, day_advice = classify_aqi(forecast_data["aqi"])
        
        with day:
            st.markdown(f"""
                <div class="forecast-card" style="border-color: {clr}40;">
                    <h4>{forecast_data['full_date']}</h4>
                    <div class="forecast-aqi" style="color: {clr};">{forecast_data['aqi']:.0f}</div>
                    <div class="forecast-label" style="color: {clr};">{lbl}</div>
                    <p style="margin: 15px 0 0 0; font-size: 0.85em; opacity: 0.8;">
                        {day_advice[:60]}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast Chart
    st.markdown("### 📉 Prediction Trend")
    forecast_df = pd.DataFrame([{"Date": d["full_date"], "AQI": d["aqi"]} for d in forecast])
    
    fig = px.bar(
        forecast_df,
        x="Date",
        y="AQI",
        color="AQI",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ff6b35", "#ef4444", "#a855f7"],
        range_color=[0, 300],
        height=400,
        title="Next 72 Hours - Predicted AQI Values with Health Thresholds"
    )
    
    fig.add_hline(y=100, line_dash="dash", line_color="#f59e0b", line_width=2,
                  annotation_text="Moderate (100)", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="Unhealthy (150)", annotation_position="right")
    
    fig.update_layout(
        hovermode="x unified",
        paper_bgcolor="rgba(15, 12, 41, 0.5)",
        plot_bgcolor="rgba(255, 255, 255, 0.05)",
        font={"color": "#ffffff"},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridwidth": 1, "gridcolor": "rgba(255, 255, 255, 0.1)"},
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Health Alerts
    st.markdown("### ⚠️ Health Alerts & Recommendations")
    
    max_forecast = max(d["aqi"] for d in forecast)
    
    if max_forecast > 150:
        alert_class = "alert-box-danger"
        alert_msg = "🚨 **UNHEALTHY CONDITIONS FORECASTED** - Avoid outdoor activities, use N95 masks, keep windows closed."
    elif max_forecast > 100:
        alert_class = "alert-box-warning"
        alert_msg = "⚠️ **MODERATE TO UNHEALTHY CONDITIONS** - Sensitive groups should limit outdoor exposure. Consider masks."
    else:
        alert_class = "alert-box-good"
        alert_msg = "✅ **GOOD AIR QUALITY** - Outdoor activities should be safe for most people."
    
    st.markdown(f'<div class="info-box {alert_class}">{alert_msg}</div>', unsafe_allow_html=True)

    if os.path.exists("artifacts/shap_summary.png"):
        st.markdown("---")
        st.markdown("### 🔍 AI Model Explainability (SHAP)")
        st.markdown("*How different factors influence our AQI predictions*")
        st.image("artifacts/shap_summary.png", caption="Feature Importance Analysis", use_column_width=True)

    st.markdown("---")

    # Footer
    st.markdown("""
        <div class="footer">
            <p><strong>Islamabad AQI Predictor</strong> | AI-Powered Environmental Intelligence</p>
            <p>
                📡 Data: AQICN (Ground Sensors) | Open-Meteo (Weather) | Hopsworks (ML Feature Store)<br>
                🤖 Models: Random Forest, Ridge Regression | 📊 Explainability: SHAP Values<br>
                👨‍💻 Developed by Zeeshan | Updated Daily at 2 AM UTC
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
