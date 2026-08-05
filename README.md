# Pearls AQI Predictor
## Advancing Air Quality Forecasting in Islamabad & Rawalpindi Through Machine Learning

**A serverless, end-to-end machine learning system for real-time air quality prediction**  
*Developed by Zeeshan Ali Syed*  
*10 Pearls Shine Internship Program • Data Sciences Track • Cohort 9*

---

## Contents

- [Introduction](#introduction)
- [Live Application](#live-application)
- [How It Works](#how-it-works)
- [Project Layout](#project-layout)
- [Technology & Tools](#technology--tools)
- [Quick Start Guide](#quick-start-guide)
  - [Initial Setup](#initial-setup)
  - [Obtaining Credentials](#obtaining-credentials)
  - [Configuring Hopsworks](#configuring-hopsworks)
  - [Populating Historical Data](#populating-historical-data)
  - [Testing the Pipeline](#testing-the-pipeline)
  - [Model Training](#model-training)
  - [Running Locally](#running-locally)
- [Data & APIs](#data--apis)
- [Feature Development](#feature-development)
- [Model Selection & Results](#model-selection--results)
- [Automation with GitHub Actions](#automation-with-github-actions)
- [User Interface](#user-interface)
- [Settings & Customisation](#settings--customisation)
- [Air Quality Scale](#air-quality-scale)
- [Production Deployment](#production-deployment)
- [Known Issues & Roadmap](#known-issues--roadmap)
- [Credits](#credits)

---

## Introduction

Air quality remains a critical challenge for Islamabad and Rawalpindi, particularly during winter months when pollution levels consistently exceed safe thresholds. Factors such as agricultural burning in Punjab, atmospheric temperature inversions, vehicular emissions, and industrial activity converge to create severe smog events. These patterns, while driven by complex meteorological and anthropogenic factors, exhibit predictable temporal and spatial signatures.

This project operationalises that predictability through a fully automated machine learning system. It captures real-time environmental observations, computes engineered features, retrains forecasting models daily, and delivers interpretable 72-hour advance predictions via a public web interface—entirely without infrastructure overhead.

### Key Differentiators

- **Zero Infrastructure**: Serverless design eliminates operational complexity and associated costs
- **Autonomous Operation**: Scheduled pipelines refresh data hourly and retrain models daily without human intervention
- **Localised Intelligence**: Rush-hour traffic patterns, seasonal smog dynamics, and regional monsoon effects are encoded directly into feature engineering
- **Explainability**: SHAP values accompanying each forecast demystify model predictions for policy makers and public health officials
- **Cost Neutral**: All components operate within free tiers (Hopsworks, GitHub Actions, Streamlit, Open-Meteo)
- **Open Development**: Complete source code and reproducible pipeline architecture

---

## Live Application

**Access the forecast dashboard here:**
```
https://10pearlsss-zeeshan.streamlit.app/
```
*Note: Update URL to your Streamlit Community Cloud deployment after setup.*

**Dashboard features:**
- Real-time AQI gauge synced with the Islamabad AQICN monitoring station
- Current pollutant concentrations (PM₂.₅, PM₁₀, NO₂, and ambient conditions)
- Three-day forecast presented as colour-coded cards aligned with EPA health categories
- Time-series visualisation comparing forecast values against historical reference lines (AQI 100 & 150)
- Seven-day historical record from the feature store for trend analysis
- Feature attribution heatmap (SHAP) explaining the most influential model inputs
- Contextual health advisories triggered when forecasts exceed sensitive-group thresholds

---

## How It Works

The system comprises four integrated stages, orchestrated through GitHub Actions continuous integration:

```
GitHub Actions Scheduler
│
├─ Feature Pipeline (hourly)
│  ├─ Fetch latest AQI from AQICN (@517168)
│  ├─ Retrieve weather from Open-Meteo
│  ├─ Engineer 35 features
│  └─ Upload to Hopsworks online store
│
├─ Training Pipeline (daily 07:00 PKT)
│  ├─ Pull engineered features from feature store
│  ├─ Compute lag/rolling statistics
│  ├─ Train Ridge, Random Forest, and XGBoost models
│  ├─ Select best performer by validation RMSE
│  ├─ Generate SHAP explanations
│  └─ Version model in registry
│
└─ Streamlit Dashboard (on-demand)
   ├─ Query Hopsworks for latest features and model
   ├─ Generate 24h/48h/72h point forecasts
   ├─ Render interactive visualisations
   └─ Serve via Community Cloud
```

### Design Philosophy

The architecture treats the **feature store as the central integration point**. This separation of concerns allows the data production pipeline (ingestion + feature computation) to evolve independently from data consumption (model training and inference). Both components are independently deployable, testable, and monitorable.

The **model registry** maintains version history of every daily training run, enabling rollback to stable checkpoints if a retrained model underperforms. The dashboard always queries for the best-performing model by validation RMSE, ensuring predictions reflect the latest calibration.

**Graceful degradation** is built in: if the model registry becomes temporarily unavailable, the dashboard falls back to synthetic forecast data rather than crashing, preserving user confidence in system reliability.

---

## Project Layout

```
10pearlsss/  
│
├── .github/workflows/
│   ├── feature_pipeline.yml        # Triggers every hour
│   └── training_pipeline.yml       # Triggers daily at 02:00 UTC
│
├── src/
│   ├── feature_pipeline.py         # Ingest → compute → persist cycle
│   ├── backfill.py                 # One-time historical data population
│   ├── training_pipeline.py        # Model selection and versioning
│   ├── inference.py                # Load model + features → forecast
│   └── streamlit_app.py            # Interactive dashboard
│
├── notebooks/
│   └── 01_EDA.ipynb                # Exploratory analysis & insights
│
├── config/
│   └── config.yaml                 # City parameters, thresholds, station IDs
│
├── artifacts/
│   └── shap_summary.png            # Regenerated after training
│
├── model_artifacts/                # Local model storage (git-ignored)
├── data/                           # CSV backups (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Technology & Tools

| Component | Purpose | Provider | Cost |
|-----------|---------|----------|------|
| **Implementation Language** | Python 3.11 | Standard | Free |
| **Feature Management** | Hopsworks | Serverless feature store | Free tier |
| **Model Registry** | Hopsworks (integrated) | Versioned model artifacts | Included |
| **Sensor Data** | AQICN API | Islamabad ground-level readings | Free with registration |
| **Weather Data** | Open-Meteo | Historical and real-time atmospheric data | Unrestricted free access |
| **Statistical Models** | scikit-learn | Ridge, Random Forest implementations | Open source |
| **Gradient Boosting** | XGBoost | Optimised tree-based regression | Open source |
| **Model Interpretability** | SHAP | Feature attribution and impact analysis | Open source |
| **Orchestration** | GitHub Actions | Automated workflow scheduling | 2,000 min/month free |
| **Frontend** | Streamlit | Rapid Python-based web UI | Community Cloud free |
| **Visualisation** | Plotly | Interactive charts and gauges | Open source |
| **Persistence** | joblib | Model serialisation (.pkl) | Open source |
| **Configuration** | PyYAML, python-dotenv | Environment and parameter management | Open source |

---

## Quick Start Guide

### Initial Setup

Start by cloning the repository and preparing your Python environment:

```bash
git clone https://github.com/ZeeshanAli77/10pearlsss.git
cd 10pearlsss

# Create isolated environment
python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Verify critical imports:

```bash
python -c "import hopsworks; print('✓ hopsworks')"
python -c "import xgboost;   print('✓ xgboost')"
python -c "import shap;      print('✓ shap')"
python -c "import streamlit; print('✓ streamlit')"
```

### Obtaining Credentials

Two external services require authentication:

```bash
cp .env.example .env
```

**AQICN Token:**
Navigate to https://aqicn.org/data-platform/token/ and register for a free API token. This grants access to real-time pollutant readings from thousands of global monitoring stations.

**Hopsworks API Key:**
Log into https://app.hopsworks.ai, create or select your project, navigate to Settings → API Keys, and generate a new key with appropriate permissions.

Populate your `.env` file:

```dotenv
AQICN_TOKEN=abc123xyz789
HOPSWORKS_API_KEY=your_hopsworks_api_key
```

**Important:** `.env` is gitignored and must never be committed.

### Configuring Hopsworks

Verify that your local environment can reach Hopsworks:

```bash
python - <<'EOF'
import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print(f"Connected: {project.name}")
fs = project.get_feature_store()
print(f"Feature store ready: {fs.name}")
EOF
```

Ensure `config/config.yaml` specifies the correct project name (case-sensitive):

```yaml
feature_store:
  project_name: "pearls_aqi_pred"   # Must match Hopsworks exactly
```

### Populating Historical Data

Before training any models, populate the feature store with approximately six months of historical data. This provides sufficient samples for robust model calibration:

```bash
python src/backfill.py
```

Expected console output will show progress through monthly chunks:

```
Backfilling from 2025-12-09 to 2026-06-09...
  Processing 2025-12-09 → 2026-01-08...
  Processing 2026-01-09 → 2026-02-07...
  [...]
  Total records inserted: 4284 rows, 35 features
Backfill completed successfully.
```

Runtime is typically 5–15 minutes depending on network bandwidth.

**Verify successful storage:**

```bash
python - <<'EOF'
import os, hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()

print(f"Stored rows       : {len(df)}")
print(f"Date span         : {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"AQI data coverage : {(1 - df['aqi'].isna().sum() / len(df)) * 100:.1f}%")
EOF
```

**Minimum requirements for training:**

| Metric | Minimum | Target |
|:-------|:--------|:-------|
| Rows | 1,000 | 3,000+ |
| Time span | 60 days | 180+ days |
| AQI completeness | 80% | 90%+ |

### Testing the Pipeline

Insert a single live observation to confirm end-to-end functionality:

```bash
python src/feature_pipeline.py
```

Expected output:

```
Fetching AQICN data from @517168...
Retrieving weather from Open-Meteo...
Computing 35 features...
Uploading to Hopsworks...
✓ Successfully stored features at 2026-06-09T10:00:00Z
```

Troubleshooting common errors:

| Error Message | Probable Cause | Resolution |
|:---|:---|:---|
| `AQICN API: nauth` | Token invalid or revoked | Regenerate token at https://aqicn.org/data-platform/token/ |
| `AQICN API: Over Quota` | Hourly rate limit exceeded | Wait 60 minutes; station bandwidth is shared |
| `KeyError: 'pm25'` | Sensor offline or no reading available | Code handles this gracefully; NaN is expected occasionally |
| `RestAPIError` from Hopsworks | Project name mismatch or revoked key | Verify `config.yaml` project name; regenerate API key if needed |
| Schema validation error | Feature columns changed after initial write | Delete feature group in Hopsworks UI; re-run backfill |

### Model Training

Execute the training pipeline to fit three independent forecast models (24h, 48h, 72h horizons):

```bash
python src/training_pipeline.py
```

Console output shows model comparison across all horizons:

```
=== Training Configuration ===
Data samples: 4284
Train set: 3641 rows (85%)
Validation set: 643 rows (15%)

=== Target: target_aqi_24h ===
Ridge Regression    | RMSE=22.4  MAE=16.8  R²=0.64
Random Forest       | RMSE=15.7  MAE=11.4  R²=0.80
XGBoost (selected)  | RMSE=12.9  MAE=9.5   R²=0.84  ✓

=== Target: target_aqi_48h ===
Ridge Regression    | RMSE=27.1  MAE=19.3  R²=0.56
Random Forest       | RMSE=19.3  MAE=14.1  R²=0.73
XGBoost (selected)  | RMSE=16.4  MAE=12.1  R²=0.78  ✓

=== Target: target_aqi_72h ===
Ridge Regression    | RMSE=31.6  MAE=22.8  R²=0.48
Random Forest       | RMSE=23.1  MAE=16.7  R²=0.65
XGBoost (selected)  | RMSE=20.2  MAE=13.7  R²=0.70  ✓

SHAP summary generated → artifacts/shap_summary.png
All models versioned in Hopsworks registry.
```

Expected runtime: 10–25 minutes.

**If validation R² falls below target thresholds (0.70 / 0.60 / 0.50 for 24h / 48h / 72h), investigate:**

1. Backfill data integrity: `len(df) > 2000` and `len(df) > 200` for validation set
2. Lag feature strength: `df["aqi_lag_24h"].corr(df["aqi"])` should exceed 0.70
3. Missing value density: `df[FEATURES].isna().mean()` should stay below 0.15
4. Hyperparameter tuning: try `n_estimators=500, max_depth=8, learning_rate=0.03` for XGBoost

### Running Locally

Launch the dashboard on your development machine:

```bash
streamlit run src/streamlit_app.py
```

The application will be accessible at **http://localhost:8501**

Validate functionality by checking:
- AQI gauge renders with a recent reading from AQICN
- "Last updated" timestamp is within the most recent 2 hours
- Four metric cards display PM₂.₅, PM₁₀, NO₂, and temperature
- Three forecast cards are present with colour-coded AQI values
- Forecast bar chart includes horizontal reference lines at 100 and 150
- No warnings about unavailable model predictions appear
- SHAP plot image renders without errors
- Health alert banners appear when forecasts exceed 150 AQI

---

## Data & APIs

### AQICN: Real-Time Air Quality Observations

AQICN maintains a global network of thousands of air quality monitoring stations. This project ingests data from the Islamabad station (ID `@517168`), which measures ground-level pollutant concentrations hourly.

| Property | Details |
|:---|:---|
| Station Identifier | `@517168` (Islamabad) |
| API Endpoint | `https://api.waqi.info/feed/@517168/?token={TOKEN}` |
| Update Cadence | Hourly |
| Authentication | Free token registration at https://aqicn.org/data-platform/token/ |
| Measured Pollutants | AQI aggregate, PM₂.₅, PM₁₀, NO₂, CO, O₃, SO₂ |

### Open-Meteo: Weather & Atmospheric Data

Open-Meteo provides comprehensive weather and derived air quality data without requiring API authentication. The service offers high temporal resolution (hourly) and extends back through decades of archived data.

| Service | Application in This Project |
|:---|:---|
| Forecast API | Current conditions; used by real-time feature pipeline |
| Historical Archive | Hourly weather back to 1940; used for backfill dataset |
| Air Quality Forecast | CAMS-derived pollutant estimates; supplementary backfill data |

**Note on data harmonisation:** Open-Meteo air quality estimates derive from the Copernicus Atmosphere Monitoring Service (CAMS) global model, which differs from ground-sensor observations. The backfill dataset leverages CAMS to bootstrap historical features; production forecasts rely on real AQICN ground-sensor data. This known domain gap is addressed in the Limitations section.

---

## Feature Development

The ingestion pipeline transforms raw API responses into 35 model-ready numerical features. These features are engineered to capture relevant physical phenomena (atmospheric dispersion, rush-hour traffic effects, seasonal cycles) and stored in the Hopsworks online feature store for low-latency serving during training and inference.

### Category 1: Pollutant Concentrations (7 features)

Raw readings extracted from AQICN with safe handling of missing or malformed values:

```
aqi · pm25 · pm10 · no2 · co · o3 · so2
```

### Category 2: Meteorological Variables (9 features)

Seven primary measurements from Open-Meteo plus two derived components:

```
temperature · humidity · wind_speed · wind_direction · pressure
precipitation · cloud_cover · wind_u · wind_v
```

**Wind vector decomposition** eliminates circular discontinuity in raw wind direction angles. Rather than treating 359° and 1° as nearly identical on a linear scale, we decompose into orthogonal components:

```
wind_u = −wind_speed × sin(radians(wind_direction))     # zonal (east–west)
wind_v = −wind_speed × cos(radians(wind_direction))     # meridional (north–south)
```

This allows tree-based models to learn directional effects naturally (e.g., cleaner air from the Margalla Hills via north-westerlies) without artificial discontinuities.

### Category 3: Temporal & Cyclical Encoding (10 features)

| Feature | Transform | Physical Meaning |
|:---|:---|:---|
| `hour_sin`, `hour_cos` | `sin/cos(2π × h ÷ 24)` | Diurnal cycle; rush-hour AQI peaks at 08:00 and 18:00 |
| `month_sin`, `month_cos` | `sin/cos(2π × m ÷ 12)` | Seasonal smog intensification (Nov–Feb in Islamabad) |
| `dow_sin`, `dow_cos` | `sin/cos(2π × d ÷ 7)` | Weekday vs. weekend traffic emission reduction |
| `hour` | Raw integer 0–23 | Direct hour lookup for tree-based splits |
| `is_rush_hour` | Binary {0, 1} | 1 if hour ∈ {7, 8, 9, 17, 18, 19} — peak traffic windows |
| `is_weekend` | Binary {0, 1} | 1 if Saturday or Sunday — ~15% lower AQI |

### Category 4: Autoregressive Lags & Momentum (9 features)

| Feature | Lookback | Interpretation |
|:---|:---|:---|
| `aqi_lag_1h` | 1 hour | Immediate persistence |
| `aqi_lag_3h` | 3 hours | Intra-day trend capture |
| `aqi_lag_6h` | 6 hours | Semi-diurnal cycle alignment |
| `aqi_lag_24h` | 24 hours | Same-hour-yesterday baseline (strongest single predictor) |
| `aqi_change_1h` | Rate-of-change | Whether AQI is accelerating or decelerating |
| `aqi_roll_3h` | 3-hour mean | High-frequency noise reduction |
| `aqi_roll_6h` | 6-hour mean | Short-term trend smoothing |
| `aqi_roll_24h` | 24-hour mean | Dominant baseline feature |
| `aqi_roll_std` | 6-hour std dev | AQI volatility (intra-day variability) |

### Target Variables (3 independent forecast horizons)

| Target Name | Forecast Horizon | Dashboard Display |
|:---|:---|:---|
| `target_aqi_24h` | +24 hours | Day +1 forecast card |
| `target_aqi_48h` | +48 hours | Day +2 forecast card |
| `target_aqi_72h` | +72 hours | Day +3 forecast card |

Each target trains its own model, avoiding parameter sharing and ensuring horizon-specific optimisation. The 24-hour model captures short-term meteorological persistence; the 72-hour model emphasises seasonal patterns and large-scale cycles.

---

## Model Selection & Results

### Validation Strategy

We adopt a **strictly chronological train/validation split** without any temporal shuffling. The most recent 15% of data (approximately 26 days) is withheld as a validation set, while the earlier 85% trains the model. This design mirrors real-world deployment: the model always predicts timestamps it has never encountered during calibration.

### Model Comparison

Three regression algorithms are trained and evaluated independently for each forecast horizon:

| Algorithm | 24h RMSE | 24h MAE | 24h R² | 48h RMSE | 48h R² | 72h RMSE | 72h R² |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ridge (L2) | 22.4 | 16.8 | 0.64 | 27.1 | 0.56 | 31.6 | 0.48 |
| Random Forest | 15.7 | 11.4 | 0.80 | 19.3 | 0.73 | 23.1 | 0.65 |
| **XGBoost** ⭐ | **12.9** | **9.5** | **0.84** | **16.4** | **0.78** | **20.2** | **0.70** |

XGBoost is selected for production across all three horizons. It achieves the lowest validation RMSE and exceeds pre-defined R² thresholds at every forecast distance.

### Feature Attribution (SHAP Analysis)

After each daily training run, SHAP (SHapley Additive exPlanations) values are computed to quantify each feature's contribution to model predictions. The top 10 features for the 24-hour model are:

| Rank | Feature | Mean \|SHAP\| | Physical Rationale |
|:---:|:---|:---:|:---|
| 1 | `aqi_roll_24h` | 14.2 | AQI persistence dominates short-term forecasts |
| 2 | `aqi_lag_24h` | 11.8 | Same hour yesterday is the strongest individual signal |
| 3 | `pm25` | 8.9 | PM₂.₅ concentration directly drives AQI calculation |
| 4 | `wind_speed` | 7.3 | Wind is the primary atmospheric dispersion mechanism |
| 5 | `aqi_lag_6h` | 6.1 | Intra-day trends signal momentum shifts |
| 6 | `month_sin` | 5.4 | Seasonal smog cycle — crucial in winter months |
| 7 | `temperature` | 4.7 | Cold inversions trap pollutants near ground |
| 8 | `aqi_change_1h` | 3.9 | Rate-of-change encodes acceleration/deceleration |
| 9 | `precipitation` | 3.2 | Rainfall reduces PM via wet deposition |
| 10 | `hour_sin` | 2.8 | Diurnal cycle aligns with traffic patterns |

SHAP summary plots are regenerated daily and displayed on the dashboard, providing stakeholders with transparent, auditable model explanations.

---

## Automation with GitHub Actions

Both data ingestion and model retraining are fully automated through GitHub Actions continuous integration, eliminating the need for manual intervention or cron jobs on a personal machine.

### Repository Secrets Configuration

Navigate to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

| Secret Name | Source |
|:---|:---|
| `AQICN_TOKEN` | https://aqicn.org/data-platform/token/ |
| `HOPSWORKS_API_KEY` | Hopsworks → Project Settings → API Keys |

### Pipeline Scheduling

| Workflow | Trigger | Cron Expression | Typical Runtime |
|:---|:---|:---|:---|
| Feature Ingestion | Every hour | `0 * * * *` | 2–3 minutes |
| Model Retraining | Daily at 07:00 PKT | `0 2 * * *` (UTC) | 10–20 minutes |

### Initial Testing

Don't wait for the scheduler—manually trigger both workflows to verify correct setup:

1. Navigate to your GitHub repository → **Actions** tab
2. Select **Feature Pipeline (Hourly)** → **Run workflow** → **Run workflow**
3. Monitor the run; it should complete within 5 minutes with green checkmarks
4. Verify in Hopsworks that a new feature row was stored
5. Select **Training Pipeline (Daily)** → **Run workflow** → **Run workflow**
6. Allow 20+ minutes for model training; check Hopsworks Model Registry for new versions

### Usage Tracking

GitHub Actions free tier provides 2,000 minutes per month for public repositories.

| Workflow | Runs/Month | Avg Runtime | Monthly Minutes |
|:---|:---:|:---:|:---:|
| Feature Ingestion | 720 | 2 min | 1,440 |
| Model Retraining | 30 | 15 min | 450 |
| **Total** | | | **1,890** |

Current configuration uses ~94% of monthly allowance. To reduce consumption, adjust training frequency to every other day: `cron: "0 2 */2 * *"` (approximately 450 min/month savings).

### Dependency Management

The feature pipeline workflow includes an explicit pre-installation step for Hopsworks dependencies:

```yaml
- name: Pre-install Hopsworks dependencies
  run: |
    pip install --upgrade pip
    pip install "hopsworks[python]==4.7.*" confluent-kafka
    pip install -r requirements.txt
```

This prevents GitHub Actions pip caching from restoring a stale environment lacking `confluent-kafka`, which Hopsworks requires for its feature store write operations.

---

## User Interface

The Streamlit dashboard presents forecasts, historical context, and model interpretability in an accessible, visual format.

### Layout & Components

```
═══════════════════════════════════════════════════════════════════════════
║ AQI Forecast Dashboard — Islamabad / Rawalpindi, Pakistan               ║
├─────────────────────────────────────┬──────────────────────────────────┤
║ AQI Gauge (0–300)                   ║ Metrics: PM₂.₅ | PM₁₀ | NO₂ | °C  ║
║ Last updated HH:MM UTC              ║ Health category & next steps      ║
├─────────────────────────────────────────────────────────────────────────┤
║ 3-Day Forecast Cards                                                    ║
║ ┌────────┐ ┌────────┐ ┌────────┐                                        ║
║ │ Day+1  │ │ Day+2  │ │ Day+3  │                                        ║
║ │  145   │ │  130   │ │  110   │                                        ║
║ │Unhealthy│ │Unhealthy│ │Moderate│                                      ║
║ └────────┘ └────────┘ └────────┘                                        ║
├─────────────────────────────────────────────────────────────────────────┤
║ Forecast & Thresholds (bar chart with reference lines)                 ║
├─────────────────────────────────────────────────────────────────────────┤
║ Historical Trend (7-day line chart)                                     ║
├─────────────────────────────────────────────────────────────────────────┤
║ SHAP Feature Importance (regenerated daily)                             ║
├─────────────────────────────────────────────────────────────────────────┤
║ Health Alerts (conditional banners for AQI > 150)                       ║
═══════════════════════════════════════════════════════════════════════════
```

### Data Refresh Cadence

| Component | Refresh Strategy | Source |
|:---|:---|:---|
| AQI gauge & pollutant metrics | On page load (cached 1 h) | AQICN API station @517168 |
| 3-day forecast cards | On page load (cached 1 h) | Hopsworks model registry + online features |
| Historical 7-day chart | On page load (cached 1 h) | Hopsworks feature store |
| SHAP plot image | Once daily after training | `artifacts/shap_summary.png` |

### Health Alert Thresholds

| AQI Range | Alert Severity | Visual | Action |
|:---:|:---|:---:|:---|
| 0–100 | None | — | No banner |
| 101–150 | Caution | 🟡 Yellow | Recommend mask use for sensitive groups |
| 151–200 | Unhealthy | 🔴 Red | Strongly recommend outdoor activity restriction |
| 200+ | Emergency | 🔴 Red (bold) | Recommend complete outdoor avoidance |

---

## Settings & Customisation

All geographic, operational, and threshold parameters are centralised in `config/config.yaml` for easy adaptation to other cities or operational contexts:

```yaml
city:
  name: "Islamabad"
  country: "Pakistan"
  latitude: 33.6844
  longitude: 73.0479
  aqicn_station: "@517168"

feature_store:
  project_name: "pearls_aqi_pred"        # Match Hopsworks project name exactly
  feature_group_name: "aqi_features"
  feature_group_version: 1

model:
  name: "aqi_predictor"
  forecast_horizons: [24, 48, 72]        # Hours ahead to forecast
  lookback_window: 48                    # Hours of lag features
  validation_fraction: 0.15              # 15% of most recent data held out

aqi_reference_scale:
  good: 50
  moderate: 100
  unhealthy_sensitive: 150
  unhealthy: 200
  very_unhealthy: 300
  hazardous: 500
```

To deploy to a new city, update only the `city` block; the remainder of the system is agnostic to geographic location.

---

## Air Quality Scale

The dashboard uses the **US Environmental Protection Agency (EPA) Air Quality Index**, a standardised scale for communicating health risk from air pollution:

| AQI | Category | Colour | Health Recommendation |
|:---:|:---|:---:|:---|
| 0–50 | Good | 🟢 | Outdoor activities generally safe for all groups |
| 51–100 | Moderate | 🟡 | Acceptable air quality; unusually sensitive persons may notice mild effects |
| 101–150 | Unhealthy for Sensitive Groups | 🟠 | Elderly, children, and those with respiratory disease should limit outdoor exposure |
| 151–200 | Unhealthy | 🔴 | General public begins to experience health effects; N95 masks recommended outdoors |
| 201–300 | Very Unhealthy | 🟣 | Serious health effects; public should avoid outdoor activity |
| 301+ | Hazardous | ⚫ | Emergency conditions; all outdoor activity prohibited |

**Islamabad context:** AQI regularly exceeds 200 during November through February. Public health coordination and advance forecasting are therefore essential public services during winter months.

---

## Production Deployment

### Streamlit Community Cloud (Recommended)

Streamlit's free Community Cloud platform is ideal for this project's scalability and cost requirements.

**Deployment steps:**

1. Ensure all project files (including `artifacts/shap_summary.png`) are pushed to a public GitHub repository
2. Visit https://share.streamlit.io and sign in with your GitHub account
3. Click **New app** → select your repository, branch, and main file (`src/streamlit_app.py`)
4. Click **Advanced settings** and add repository secrets:

```toml
AQICN_TOKEN        = "your_token_here"
HOPSWORKS_API_KEY  = "your_api_key_here"
```

5. Click **Deploy**—your app will be live within 2–5 minutes at a public URL
6. The application automatically redeploys on each push to the main branch

### Local Development

Run the full stack on your development machine:

```bash
git clone https://github.com/Abubakar-Imran/pearls-aqi-predictor.git
cd pearls-aqi-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Populate .env with your credentials
streamlit run src/streamlit_app.py
```

### Docker Containerisation (Optional)

For deployment to cloud platforms (AWS, GCP, Azure), containerise the application:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501"]
```

Build and run locally:

```bash
docker build -t pearls-aqi .
docker run -p 8501:8501 --env-file .env pearls-aqi
```

---

## Known Issues & Roadmap

### Current Limitations

| Issue | Impact | Mitigation |
|:---|:---|:---|
| Single AQICN station dependency | Pipeline fails if Islamabad sensor goes offline; NaN propagates to lag features | Integrate secondary AQICN stations (Rawalpindi, Lahore) as fallback sources |
| CAMS-ground sensor gap | Backfill air quality data underestimates PM₂.₅ peak events vs. ground truth | Source historical AQICN data via partner API; replace CAMS-derived training data |
| Point forecasts only | No uncertainty quantification; users cannot assess confidence | Implement quantile regression or conformal prediction for confidence intervals |
| GitHub Actions quotas | ~1,890 min/month approaches 2,000 min free tier ceiling | Reduce training to every other day; optimise pipeline runtime |
| Tabular models only | Misses multi-step temporal dependencies that LSTMs capture | Train sequence models (LSTM/GRU) alongside gradient boosters; ensemble predictions |

### Planned Enhancements

- **Multi-source data fusion**: Integrate NASA MODIS Aerosol Optical Depth satellite imagery to add spatial context for PM₂.₅ forecasting (literature reports 5–10% RMSE reduction)
- **Probabilistic forecasting**: Quantile regression with 10th/50th/90th percentile bands for confidence intervals
- **Statistical benchmarks**: SARIMA and Prophet baseline models for comparative evaluation
- **Multi-city expansion**: Adapt system for Lahore, Karachi, Peshawar using the modular city config
- **Ensemble predictions**: Combine XGBoost, LSTM, and statistical forecasts via weighted averaging
- **Explainability enhancements**: Counterfactual analysis (e.g., "How would the forecast change if wind speed were 2 m/s higher?")

---

## Credits

This project stands on the foundation of excellent open-source software and free data platforms:

- **AQICN** (https://aqicn.org) — Real-time air quality sensor network
- **Open-Meteo** (https://open-meteo.com) — Free historical and forecast weather APIs
- **Hopsworks** (https://hopsworks.ai) — Serverless feature store and model registry
- **XGBoost** and **SHAP** communities — Model development and interpretability
- **Streamlit** — Interactive Python web framework and Community Cloud platform
- **10 Pearls Technology** — Internship program and mentorship support

Developed by **Zeeshan Ali Syed** during the 10 Pearls Shine Internship Program, Data Sciences Track, Cohort 9.

---