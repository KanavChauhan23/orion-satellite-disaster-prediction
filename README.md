# 🛰 ORION — AI-Driven Satellite Swarm for Autonomous Disaster Prediction
<div align="center">

![Status](https://img.shields.io/badge/Status-Prototype-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-API-black?style=flat-square&logo=flask)
![AI](https://img.shields.io/badge/AI-Scikit--Learn-orange?style=flat-square&logo=scikitlearn)
![Satellite](https://img.shields.io/badge/Domain-Satellite%20AI-purple?style=flat-square)

**AI-powered satellite swarm system that predicts natural disasters using multi-spectral satellite data and machine learning.**

**Live Demo → [ORION Mission DP](https://orion-satellite-disaster-prediction-77gvxe6l1.vercel.app/)**

</div>

---

## 🌍 Project Overview

**ORION** is an AI-based disaster prediction prototype designed to simulate how a swarm of Earth-observation satellites can analyze environmental signals and detect early indicators of natural disasters.

The system integrates **satellite sensor features, atmospheric indicators, and machine learning models** to identify risks of events such as **floods, wildfires, earthquakes, cyclones, and droughts**.

This project demonstrates how **AI + satellite data + autonomous swarm systems** could help governments and disaster response agencies detect threats earlier and improve global disaster preparedness.

---

## ⚡ Key Capabilities

| Feature | Description |
|-------|-------------|
| 🛰 **Satellite Swarm Simulation** | Simulates multiple satellites collecting environmental data |
| 🤖 **AI Disaster Prediction** | Machine learning model predicts disaster type and severity |
| 🌍 **Mission Control Dashboard** | Interactive interface to monitor satellite activity |
| 📊 **Environmental Data Analysis** | Processes 14 atmospheric and terrain features |
| ⚠ **Real-time Alert System** | Generates alerts with confidence scores |
| 📈 **Risk Monitoring** | Tracks disaster trends and alert history |
| 🌐 **API-Driven Architecture** | REST API for prediction and satellite data |

---

## 🛠 Tech Stack

### Backend
- **Python**
- **Flask** — REST API
- **Scikit-learn** — Machine learning model
- **NumPy / Pandas** — Data processing

### Frontend
- **HTML5**
- **JavaScript**
- **Canvas API** — satellite visualization
- **CSS3** — mission control UI

### AI & Data
- Gradient Boosting Classifier
- Synthetic satellite sensor dataset
- Multi-spectral environmental features


---

## 📁 Project Structure

```
disaster-sat/
├── frontend/
│   └── index.html              ← Mission Control Dashboard (self-contained)
├── backend/
│   ├── app.py                  ← Flask REST API
│   └── satellite_data.py       ← Satellite data simulator
├── models/
│   └── disaster_predictor.py   ← AI prediction model (scikit-learn)
├── data/                       ← Generated outputs
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A — Browser Only (Instant Demo)
```bash
# Just open the dashboard directly:
open frontend/index.html
```
The dashboard runs entirely in the browser with a built-in simulation engine.

### Option B — Full Stack (Backend + Dashboard)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Train AI model and run data simulator
cd models
python disaster_predictor.py

# 3. Generate satellite data samples
cd ../backend
python satellite_data.py

# 4. Start Flask API
python app.py
# → API available at http://localhost:5000/api/

# 5. Open the dashboard
open ../frontend/index.html
```

---

## 🧠 AI Model Details

| Attribute         | Value                              |
|-------------------|------------------------------------|
| Algorithm         | Gradient Boosting Classifier       |
| Training samples  | 8,000 synthetic satellite readings |
| Input features    | 14 spectral / atmospheric indices  |
| Output classes    | 6 (Flood, Wildfire, Earthquake,    |
|                   |    Cyclone, Drought, No Threat)    |
| Accuracy          | ~94%                               |

### Features Used

| Feature     | Source               | Description                      |
|-------------|----------------------|----------------------------------|
| NDVI        | Sentinel-2 / MODIS   | Vegetation health index          |
| NDWI        | Sentinel-2           | Water content index              |
| LST         | MODIS / Landsat      | Land surface temperature (°C)    |
| SWIR        | Sentinel-2           | Short-wave infrared reflectance  |
| NIR         | Sentinel-2           | Near-infrared reflectance        |
| Precip      | NASA TRMM / GPM      | Precipitation anomaly (mm/day)   |
| Wind Speed  | NOAA / ERA5          | Wind speed (km/h)                |
| Humidity    | ERA5 reanalysis      | Relative humidity (%)            |
| Soil Moist  | NASA GRACE           | Soil moisture (%)                |
| SST Anomaly | NOAA GOES / Sentinel | Sea surface temperature anomaly  |
| Elevation   | SRTM DEM             | Terrain height (m)               |
| Slope       | SRTM DEM             | Terrain slope (degrees)          |
| Seismic V   | USGS SeismoNet       | Seismic velocity anomaly         |
| Cloud Cover | MODIS                | Cloud coverage (%)               |

---

## 🛰 Satellite Swarm

6 simulated satellites in Low Earth Orbit (LEO) and Sun-Synchronous Orbit (SSO):

| Satellite | Altitude | Orbit | Inclination |
|-----------|----------|-------|-------------|
| SAT-1001  | 450 km   | LEO   | 53°         |
| SAT-1002  | 620 km   | SSO   | 97.6°       |
| SAT-1003  | 510 km   | LEO   | 45°         |
| SAT-1004  | 780 km   | LEO   | 70°         |
| SAT-1005  | 430 km   | LEO   | 53°         |
| SAT-1006  | 680 km   | SSO   | 97.6°       |

---

## 🌐 API Endpoints

```
GET  /api/status            System status
GET  /api/satellites        Live satellite positions
POST /api/predict           Predict disaster from features
GET  /api/scan?n=12         Trigger global scan (n regions)
GET  /api/region/{lat}/{lon} Scan specific location
GET  /api/alerts            Alert history
GET  /api/stats             Prediction statistics
GET  /api/history           7-day alert history
```

### Example: Predict via API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ndvi": 0.05, "ndwi": -0.3, "lst": 48,
    "swir": 0.5,  "nir": 0.15,  "precip": -30,
    "wind_speed": 55, "humidity": 18, "soil_moist": 8,
    "sst_anom": 0.2,  "elevation": 400, "slope": 18,
    "seismic_v": 0.1, "cloud_cov": 10
  }'
```
Response:
```json
{
  "prediction": "Wildfire",
  "confidence": 91.4,
  "severity": "critical",
  "risk_scores": { "Wildfire": 91.4, "Drought": 5.2, ... }
}
```

---

## 🔮 Dashboard Features

- **Live Satellite Visualization** — animated orbital paths on world map
- **Real-time Alert Feed** — color-coded by severity (low → critical)
- **Manual Prediction** — input 14 satellite features and get instant AI prediction
- **5 Disaster Presets** — one-click flood/wildfire/cyclone/earthquake/drought scenarios
- **Risk Overview** — donut chart + per-disaster alert counts
- **Sparkline Metrics** — total alerts, critical count, regions scanned, avg confidence
- **Notification System** — bottom-right toast notifications
- **Auto Global Scan** — triggers every 45 seconds automatically

---

## 💡 Patent Highlights

1. **AI-driven multi-spectral analysis** — combines 14 satellite sensor types
2. **Swarm coordination** — 6 satellites with different orbits for full coverage
3. **Real-time prediction** — sub-second disaster classification
4. **Multi-hazard detection** — 5 disaster types in a single model
5. **Severity grading** — 4-tier alert system (low/moderate/high/critical)
6. **Time-series monitoring** — tracks disaster development over 30 days

---

## 🛠 Technologies

- **Python** — AI/ML (scikit-learn, numpy, pandas)
- **Flask** — REST API backend
- **JavaScript** — Browser simulation engine
- **Canvas API** — Real-time satellite visualization
- **CSS3** — Mission control dashboard design

---

## 📊 Real Data Sources (for production)

| Source              | Data Type              | URL                         |
|---------------------|------------------------|-----------------------------|
| NASA Earthdata      | MODIS, GRACE, FIRMS    | earthdata.nasa.gov          |
| Google Earth Engine | Sentinel, Landsat      | earthengine.google.com      |
| Copernicus          | Sentinel-1/2/3         | scihub.copernicus.eu        |
| NOAA GOES           | Weather, Hurricane     | noaa.gov                    |
| USGS               | Seismic, Elevation     | usgs.gov                    |

---

*Built for patent submission and prototype demonstration.*
*© 2026 — AI-Driven Satellite Swarm Project*
