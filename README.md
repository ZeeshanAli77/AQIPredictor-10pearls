# AQI Predictor: Serverless Air Quality Forecasting for Islamabad & Rawalpindi

**By Zeeshan Ali Syed** | 10 Pearls Shine Internship Program (Cohort 9, Data Sciences)

---

## The Problem

Islamabad and Rawalpindi face a recurring air quality crisis. From November through February, pollution levels consistently breach hazardous thresholds—driven by crop burning, thermal inversions, vehicular emissions, and industrial activity. Residents lack reliable advance warning.

This project operationalizes a solution: a fully automated ML system that predicts AQI up to 72 hours ahead with interpretable explanations, deployed entirely on free infrastructure.

---

## Quick Links

- **Live Dashboard**: https://10pearlsss-zeeshan.streamlit.app/
- **GitHub**: https://github.com/ZeeshanAli77/AQIPredictor-10pearls
- **Live Data**: Updated hourly; forecasts retrained daily at 07:00 PKT

---

## How It Actually Works 

Every hour, the system wakes up and does this:

1. **Asks AQICN**: "What's the current PM2.5, NO₂, and AQI at station @517168?"
2. **Asks Open-Meteo**: "What's the current wind, temperature, humidity, pressure, rainfall?"
3. **Calculates 35 features** from those readings (rolling averages, 24-hour lags, rush-hour flags, seasonal encodings)
4. **Stores in Hopsworks** (a cloud feature store) for later reuse
5. **Dashboard queries** these features plus a trained XGBoost model to generate a 3-day forecast

Every night at 02:00 UTC (07:00 PKT), a separate training job:

1. Pulls 6+ months of features from Hopsworks
2. Trains three models (Ridge, Random Forest, XGBoost) independently for 24h, 48h, and 72h horizons
3. Picks the winner (usually XGBoost; RMSE ≈ 12.9, 16.4, 20.2 for the three horizons)
4. Computes SHAP values to explain which features mattered most
5. Saves the best model and explanation plots to the model registry

All orchestrated by GitHub Actions cron jobs. **Zero infrastructure to manage.**

---

## Under the Hood: The Four Pillars

### 1️⃣ Data Ingestion (Hourly)

**AQICN API** → ground-truth pollutant readings
- Station: Islamabad (@517168)
- Pollutants: AQI, PM2.5, PM10, NO₂, CO, O₃, SO₂
- Free tier; requires registration token
- Update frequency: hourly

**Open-Meteo API** → weather + atmospheric data
- No API key required (completely free)
- Current conditions for live ingestion
- 80+ years of hourly history for backfill
- Data points: temperature, humidity, wind, pressure, clouds, rainfall

### 2️⃣ Feature Engineering (35 Features Total)

The raw numbers are useless to a model without transformation. We compute:

**Pollutant Raw Values (7)**
```
aqi, pm25, pm10, no2, co, o3, so2
```

**Weather Observations (9)**
```
temperature, humidity, wind_speed, wind_direction, pressure, precipitation, cloud_cover
+ wind_u, wind_v  (decomposed wind vectors to avoid circular discontinuity)
```

**Temporal Encodings (10)** — Because AQI has strong daily and seasonal cycles
```
hour (0-23), is_rush_hour (peak traffic windows: 7-9 AM, 5-7 PM)
is_weekend (lower AQI on weekends)
hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos (cyclical encoding)
```

**Autoregressive Features (9)** — The strongest predictors
```
aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_24h (yesterday at same hour)
aqi_change_1h (rate of change — is it rising or falling?)
aqi_roll_3h, aqi_roll_6h, aqi_roll_24h (rolling means)
aqi_roll_std (volatility over 6 hours)
```

**Target Variables (3)** — One model per horizon
```
target_aqi_24h, target_aqi_48h, target_aqi_72h
```

### 3️⃣ Model Training & Comparison (Daily)

Three algorithms are trained independently on each horizon:

| Algorithm     | 24h RMSE | 24h R² | 48h RMSE | 48h R² | 72h RMSE | 72h R² | Winner? |
|---------------|----------|--------|----------|--------|----------|--------|---------|
| Ridge         | 22.4     | 0.64   | 27.1     | 0.56   | 31.6     | 0.48   | ❌      |
| Random Forest | 15.7     | 0.80   | 19.3     | 0.73   | 23.1     | 0.65   | ⚠️      |
| XGBoost       | **12.9** | **0.84** | **16.4** | **0.78** | **20.2** | **0.70** | ✅      |

The best model is versioned and deployed; old versions are kept for rollback.

**Top 10 Most Important Features** (by SHAP):
1. `aqi_roll_24h` (24-hour rolling mean) — AQI is highly persistent
2. `aqi_lag_24h` (same hour yesterday) — strongest single predictor
3. `pm25` (raw PM2.5 concentration) — directly drives AQI
4. `wind_speed` (meteorology) — disperses pollutants
5. `aqi_lag_6h` (recent trend) — signals acceleration
6. `month_sin` (seasonal encoding) — winter smog dominance
7. `temperature` (cold inversions trap PM2.5)
8. `aqi_change_1h` (rate of change)
9. `precipitation` (wet deposition clears PM)
10. `hour_sin` (diurnal cycle / rush hours)

### 4️⃣ Dashboard & Inference (On-Demand)

Streamlit app that displays:
- **Live AQI Gauge** → synced with AQICN every 2 hours
- **Current Pollutants** → PM2.5, PM10, NO₂, temperature
- **3-Day Forecast Cards** → colour-coded by EPA health categories
- **Forecast Bar Chart** → with reference lines at AQI 100 and 150
- **7-Day Historical Trend** → from feature store
- **SHAP Attribution Plot** → explains the model's reasoning
- **Health Alerts** → conditional banners when forecast exceeds thresholds

---

## Getting Started: Step-by-Step

### Prerequisites
- Python 3.11+
- Free [Hopsworks account](https://app.hopsworks.ai) with project named `pearls_aqi_pred`
- Free [AQICN token](https://aqicn.org/data-platform/token/)
- Git

### Step 1: Clone & Setup Environment

```bash
git clone https://github.com/ZeeshanAli77/AQIPredictor-10pearls.git
cd AQIPredictor-10pearls

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```
AQICN_TOKEN=your_token_from_aqicn_dot_org
HOPSWORKS_API_KEY=your_api_key_from_hopsworks_project_settings
```

Test the connection:
```bash
python -c "
import os, hopsworks
from dotenv import load_dotenv
load_dotenv()
proj = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
print(f'✓ Connected to {proj.name}')
"
```

### Step 3: Populate Historical Data (One-Time)

Before training any models, populate 6 months of history:

```bash
python src/backfill.py
```

Expected output:
```
Backfilling from 2025-12-09 to 2026-06-09...
  [████████████████████] 100%
Total records inserted: 4284 rows, 35 features
```

Verify:
```bash
python -c "
import os, hopsworks
from dotenv import load_dotenv
load_dotenv()
proj = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
fg = proj.get_feature_store().get_feature_group('aqi_features', version=1)
print(f'Rows: {len(fg.read())}')
"
```

**Minimum requirements:**
- 1,000+ rows (preferably 3,000+)
- 60+ days span (preferably 180+)
- 80%+ AQI completeness (preferably 90%+)

### Step 4: Test the Feature Pipeline

Insert one live row:

```bash
python src/feature_pipeline.py
```

Expected:
```
Fetching AQICN data from @517168...
Retrieving weather from Open-Meteo...
Computing 35 features...
Uploading to Hopsworks...
✓ Successfully stored features at 2026-06-09T10:00:00Z
```

**Troubleshooting:**

| Error | Fix |
|-------|-----|
| `nauth` from AQICN | Regenerate token at [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `Over Quota` from AQICN | Wait 1 hour; station has hourly rate limits |
| `KeyError: 'pm25'` | Station may be offline; code handles gracefully (NaN inserted) |
| Hopsworks `RestAPIError` | Check `config.yaml` project name matches exactly (case-sensitive) |

### Step 5: Train Models

```bash
python src/training_pipeline.py
```

Expected:
```
Fetching training data (4284 samples)...
Train/val split: 3641 / 643 rows

=== target_aqi_24h ===
Ridge:       RMSE=22.4, R²=0.64
RF:          RMSE=15.7, R²=0.80
XGBoost:     RMSE=12.9, R²=0.84  ✓ BEST
SHAP plot → artifacts/shap_summary.png
Model registered: aqi_predictor_24h v1

[... 48h and 72h follow ...]
```

**Runtime:** 10–25 minutes

**If R² < 0.60:**
1. Ensure backfill has 2,000+ rows
2. Check lag correlation: `df["aqi_lag_24h"].corr(df["aqi"]) > 0.70`
3. Check NaN density: `df[features].isna().mean() < 0.15`
4. Try XGBoost tuning: `n_estimators=500, max_depth=8, learning_rate=0.03`

### Step 6: Launch Dashboard

```bash
streamlit run src/streamlit_app.py
```

Open http://localhost:8501

**Validation checklist:**
- [ ] AQI gauge shows real number (not stale)
- [ ] Pollutant cards render with values
- [ ] 3-day forecast cards show colour-coded AQI
- [ ] Bar chart displays with reference lines
- [ ] Historical 7-day chart loads
- [ ] SHAP image displays at bottom
- [ ] No error messages in console

---

## Project Architecture at a Glance

```
GITHUB ACTIONS (Orchestrator)
  │
  ├─ Feature Pipeline (hourly)
  │  └─ AQICN + Open-Meteo → compute 35 features → Hopsworks
  │
  └─ Training Pipeline (daily @ 07:00 PKT)
     └─ Hopsworks (features) → Ridge/RF/XGBoost → model registry
        
HOPSWORKS (Feature Store + Model Registry)
  │
  ├─ Online Feature Store (aqi_features)
  │  └─ 35 features/row, updated hourly
  │
  └─ Model Registry (aqi_predictor_24h / 48h / 72h)
     └─ Versioned daily; best model by RMSE served

STREAMLIT DASHBOARD (On-Demand)
  └─ Query features + model → render 3-day forecast + SHAP
```

---

## Configuration & Customization

All parameters live in `config/config.yaml`:

```yaml
ccity:
  name: "Islamabad"
  country: "Pakistan"
  latitude: 33.7298
  longitude: 73.1772
  aqicn_station: "islamabad"   

feature_store:
  project_name: "zeeshanproject"
      
  
model:
  forecast_horizons: [24, 48, 72]     # Hours ahead
  lookback_window: 48                 # Hours of lag features
  validation_fraction: 0.15           # Last 15% held out

aqi_thresholds:
  good: 50
  moderate: 100
  unhealthy_sensitive: 150
  unhealthy: 200
```

**To adapt to another city**, only change the `city` block—the rest is agnostic.

---

## Automating with GitHub Actions

Two workflows run automatically once you push and add repository secrets.

### Add Secrets to Your Repo

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|--------|-------|
| `AQICN_TOKEN` | From [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) |
| `HOPSWORKS_API_KEY` | From Hopsworks project → Settings → API Keys |

### Workflow Schedules

| Workflow | When | Cron | Runtime |
|----------|------|------|---------|
| Feature Pipeline | Every hour | `0 * * * *` | ~2–3 min |
| Training Pipeline | Daily 07:00 PKT | `0 2 * * *` (UTC) | ~15 min |

**Monthly usage:** ~1,890 min out of GitHub's 2,000 min free tier.

### Test Both Workflows Manually

Before relying on the schedule:

1. **Actions tab** → **Feature Pipeline (Hourly)** → **Run workflow**
2. Wait for green ✓, then verify a new row appeared in Hopsworks
3. **Training Pipeline (Daily)** → **Run workflow**
4. Wait 20+ min, check Model Registry for new versions

---

## EPA Air Quality Scale Reference

| AQI     | Category                       | Colour | What It Means                                                            |
|---------|--------------------------------|--------|--------------------------------------------------------------------------|
| 0–50    | Good                           | 🟢      | Air quality is satisfactory                                              |
| 51–100  | Moderate                       | 🟡      | Unusually sensitive people may notice mild effects                       |
| 101–150 | Unhealthy for Sensitive Groups | 🟠      | Children, elderly, and those with respiratory disease should limit time  |
| 151–200 | Unhealthy                      | 🔴      | Everyone may experience health effects; N95 masks recommended            |
| 201–300 | Very Unhealthy                 | 🟣      | Health warnings for the general public; avoid outdoor activity           |
| 301+    | Hazardous                      | ⚫      | Emergency conditions; all outdoor activity prohibited                    |

**For Islamabad:** AQI regularly exceeds 200 during November–February. This system enables advance planning.

---

## Deployment Options

### Option A: Streamlit Community Cloud (Easiest)

1. Push code to a public GitHub repo
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch (`main`), and file (`src/streamlit_app.py`)
4. Add secrets in the settings UI (same keys as GitHub Actions)
5. Click **Deploy** — live in 2–5 minutes
6. Auto-redeploys on every push to `main`

### Option B: Local Machine

```bash
git clone <your-repo>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your secrets
streamlit run src/streamlit_app.py
```

### Option C: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501"]
```

```bash
docker build -t pearls-aqi .
docker run -p 8501:8501 --env-file .env pearls-aqi
```

---

## Known Limitations

| Issue | Impact | Workaround |
|-------|--------|-----------|
| Single AQICN station | Pipeline fails if sensor offline | Add fallback stations (Rawalpindi sector monitors) |
| CAMS vs. ground sensors | Backfill underestimates PM2.5 peaks | Replace with historical AQICN data via partner API |
| Point forecasts only | No uncertainty quantification | Implement quantile regression for confidence bands |
| GitHub Actions quota | 1,890 min/month ≈ 94% of free tier | Reduce training to every 2 days: `0 2 */2 * *` |
| Tabular models only | Misses multi-step temporal dependencies | Add LSTM to pipeline alongside XGBoost |

---

## Roadmap

- **Multi-source integration:** Fuse AQICN, EPA-AirNow, and satellite AOD data
- **Probabilistic forecasts:** Quantile regression + conformal prediction for confidence intervals
- **Ensemble methods:** Weight Ridge, RF, XGBoost, LSTM, and SARIMA predictions
- **Explainability:** Counterfactual analysis ("How would forecast change if wind were 2 m/s higher?")
- **Multi-city expansion:** Lahore, Karachi, Peshawar (config-only changes needed)
- **Statistical baselines:** SARIMA and Prophet for comparison

---

## Project Structure

```
10PEARLSSS/10pearls-zeeshan
├── .github/workflows/
│   ├── feature_pipeline.yml          # Hourly ingestion
│   └── training_pipeline.yml         # Daily retraining
├── src/
│   ├── feature_pipeline.py           # Ingest + compute + store
│   ├── backfill.py                   # Historical population
│   ├── training_pipeline.py          # Model training + versioning
│   ├── inference.py                  # Load model + forecast
│   └── streamlit_app.py              # Interactive dashboard
├── notebooks/
│   └── 01_EDA.ipynb                  # Exploratory analysis
├── config/
│   └── config.yaml                   # All parameters
├── artifacts/
│   └── shap_summary.png              # Regenerated daily
├── data/                             # Local backups (git-ignored)
├── model_artifacts/                  # Local models (git-ignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Component | Technology | Purpose | Cost |
|-----------|-----------|---------|------|
| Language | Python 3.11 | All code | Open source |
| Feature Store | Hopsworks | Centralized feature management | Free tier |
| Model Registry | Hopsworks | Versioned models | Included |
| Data | AQICN API | Ground-truth AQI/pollutants | Free token |
| Weather | Open-Meteo | No key needed; free forever | Free |
| ML Algorithms | scikit-learn, XGBoost | Ridge, RF, XGBoost | Open source |
| Explainability | SHAP | Feature importance | Open source |
| Orchestration | GitHub Actions | Scheduled workflows | 2,000 min/month free |
| UI Framework | Streamlit | Interactive dashboard | Community Cloud free |
| Visualization | Plotly | Gauges, charts, plots | Open source |
| Serialization | joblib | Model persistence | Open source |
| Config | YAML, python-dotenv | Parameters & secrets | Open source |

---

## FAQ

**Q: How often does the forecast update?**
A: Features are ingested every hour. Models are retrained daily at 07:00 PKT. Dashboard refreshes on page load (cached 1 hour).

**Q: What if AQICN station goes offline?**
A: The `safe_extract()` function returns NaN, which propagates into lag features. The model still makes predictions (though quality degrades). Future: integrate fallback stations.

**Q: Why XGBoost over Random Forest?**
A: Gradient boosting captured non-linear interactions better. RMSE was 12.9 vs. 15.7 at 24h horizon. All three horizons favored XGBoost.

**Q: Can I use this for another city?**
A: Yes. Update `config/config.yaml` with new city coordinates, station ID, and project name. Everything else is location-agnostic.

**Q: How much does it cost to run?**
A: Zero. All components operate within free tiers (Hopsworks, GitHub Actions, Streamlit Community Cloud, Open-Meteo).

---

## Credits

Built with:
- **AQICN** for real-time sensor data
- **Open-Meteo** for free weather APIs
- **Hopsworks** for serverless feature store
- **XGBoost** & **SHAP** communities
- **Streamlit** for rapid UI development
- **10 Pearls Technology** for the internship opportunity

**Author:** Zeeshan Ali Syed  
**Program:** 10 Pearls Shine Internship (Cohort 9, Data Sciences)  
**Contact:** zeeshann.2003@gmail.com
