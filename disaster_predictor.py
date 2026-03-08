"""
AI-Driven Satellite Swarm - Disaster Prediction Model
======================================================
Simulates an AI model that analyzes satellite imagery features
to predict natural disaster risks: floods, wildfires, earthquakes,
cyclones/hurricanes, and droughts.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import os
import joblib
from datetime import datetime, timedelta
import random

# ── Disaster Types ──────────────────────────────────────────────────────────
DISASTER_TYPES = ["flood", "wildfire", "earthquake", "cyclone", "drought", "none"]

DISASTER_LABELS = {
    0: "Flood",
    1: "Wildfire",
    2: "Earthquake",
    3: "Cyclone",
    4: "Drought",
    5: "No Threat"
}

SEVERITY_LEVELS = {
    "low":      {"color": "#22c55e", "threshold": 0.3},
    "moderate": {"color": "#f59e0b", "threshold": 0.5},
    "high":     {"color": "#f97316", "threshold": 0.7},
    "critical": {"color": "#ef4444", "threshold": 0.9},
}

# ── Feature Engineering ──────────────────────────────────────────────────────
def generate_satellite_features(n_samples=5000, random_seed=42):
    """
    Simulate satellite sensor readings and derived indices.

    Features mimic real satellite bands and derived indices:
    - NDVI  : Normalized Difference Vegetation Index
    - NDWI  : Normalized Difference Water Index
    - LST   : Land Surface Temperature (°C)
    - SWIR  : Short-Wave Infrared reflectance
    - NIR   : Near-Infrared reflectance
    - Precip: Precipitation anomaly (mm/day)
    - WindSp: Wind speed (km/h)
    - Humid : Relative humidity (%)
    - SoilM : Soil moisture (%)
    - SeaSST: Sea Surface Temperature anomaly (°C)
    - Elev  : Elevation (m)
    - Slope : Terrain slope (degrees)
    - SeismV: Seismic velocity anomaly
    - CloudC: Cloud cover (%)
    """
    np.random.seed(random_seed)
    n = n_samples

    features = {
        "ndvi":       np.random.uniform(-0.2, 0.9, n),
        "ndwi":       np.random.uniform(-0.5, 0.8, n),
        "lst":        np.random.uniform(5.0, 55.0, n),
        "swir":       np.random.uniform(0.0, 0.6, n),
        "nir":        np.random.uniform(0.0, 0.7, n),
        "precip":     np.random.uniform(-50, 200, n),
        "wind_speed": np.random.uniform(0, 200, n),
        "humidity":   np.random.uniform(10, 100, n),
        "soil_moist": np.random.uniform(0, 100, n),
        "sst_anom":   np.random.uniform(-3, 5, n),
        "elevation":  np.random.uniform(0, 3000, n),
        "slope":      np.random.uniform(0, 45, n),
        "seismic_v":  np.random.uniform(-2, 2, n),
        "cloud_cov":  np.random.uniform(0, 100, n),
    }

    labels = _assign_labels(features)
    df = pd.DataFrame(features)
    df["label"] = labels
    return df


def _assign_labels(f):
    """Rule-based label assignment simulating real disaster signatures."""
    n = len(f["ndvi"])
    labels = np.full(n, 5)  # default: No Threat

    for i in range(n):
        # ── Flood: high NDWI, high precip, low NDVI, low elevation
        if (f["ndwi"][i] > 0.4 and
                f["precip"][i] > 80 and
                f["soil_moist"][i] > 70 and
                f["elevation"][i] < 500):
            labels[i] = 0

        # ── Wildfire: high LST, low NDVI, low humidity, high SWIR
        elif (f["lst"][i] > 40 and
              f["ndvi"][i] < 0.15 and
              f["humidity"][i] < 30 and
              f["swir"][i] > 0.35):
            labels[i] = 1

        # ── Earthquake: seismic velocity anomaly + slope
        elif (abs(f["seismic_v"][i]) > 1.5 and
              f["slope"][i] > 20):
            labels[i] = 2

        # ── Cyclone: high wind, high SST anomaly, high cloud cover
        elif (f["wind_speed"][i] > 120 and
              f["sst_anom"][i] > 1.5 and
              f["cloud_cov"][i] > 70):
            labels[i] = 3

        # ── Drought: low NDVI, low soil moisture, high LST, low precip
        elif (f["ndvi"][i] < 0.1 and
              f["soil_moist"][i] < 20 and
              f["lst"][i] > 38 and
              f["precip"][i] < -20):
            labels[i] = 4

    return labels


# ── Model Training ────────────────────────────────────────────────────────────
class DisasterPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = [
            "ndvi", "ndwi", "lst", "swir", "nir",
            "precip", "wind_speed", "humidity", "soil_moist",
            "sst_anom", "elevation", "slope", "seismic_v", "cloud_cov"
        ]
        self.trained = False

    def train(self, df: pd.DataFrame):
        X = df[self.feature_names].values
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        print("🚀 Training Disaster Prediction Model...")
        self.model.fit(X_train_sc, y_train)

        y_pred = self.model.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {acc*100:.2f}%")
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=[DISASTER_LABELS[i] for i in range(6)]
        ))

        self.trained = True
        return acc

    def predict(self, features: dict) -> dict:
        """Predict disaster risk from a feature dict."""
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        x = np.array([[features[f] for f in self.feature_names]])
        x_sc = self.scaler.transform(x)

        proba = self.model.predict_proba(x_sc)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # build sorted risk scores
        risk_scores = {
            DISASTER_LABELS[i]: round(float(proba[i]) * 100, 2)
            for i in range(len(DISASTER_LABELS))
        }

        severity = "low"
        for lvl, cfg in SEVERITY_LEVELS.items():
            if confidence >= cfg["threshold"]:
                severity = lvl

        return {
            "prediction":  DISASTER_LABELS[pred_class],
            "confidence":  round(confidence * 100, 2),
            "severity":    severity,
            "risk_scores": risk_scores,
            "timestamp":   datetime.utcnow().isoformat(),
        }

    def save(self, path="models/disaster_model.pkl"):
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"💾 Model saved to {path}")

    def load(self, path="models/disaster_model.pkl"):
        obj = joblib.load(path)
        self.model  = obj["model"]
        self.scaler = obj["scaler"]
        self.trained = True
        print(f"📂 Model loaded from {path}")


# ── Satellite Swarm Simulation ────────────────────────────────────────────────
class SatelliteSwarm:
    """Simulates a swarm of satellites scanning the Earth."""

    def __init__(self, n_satellites=6):
        self.n_satellites = n_satellites
        self.satellites = self._init_swarm()
        self.predictor = DisasterPredictor()
        self._train_predictor()

    def _init_swarm(self):
        sats = []
        for i in range(self.n_satellites):
            sats.append({
                "id":          f"SAT-{1001 + i}",
                "altitude_km": random.randint(400, 800),
                "inclination": random.choice([45, 53, 70, 97.6]),
                "orbit_type":  random.choice(["LEO", "SSO"]),
                "sensors":     ["optical", "SAR", "thermal", "multispectral"],
                "status":      "active",
                "lat":         random.uniform(-90, 90),
                "lon":         random.uniform(-180, 180),
                "coverage_km": random.randint(200, 600),
            })
        return sats

    def _train_predictor(self):
        df = generate_satellite_features(n_samples=8000)
        self.predictor.train(df)

    def scan_region(self, lat: float, lon: float) -> dict:
        """Simulate scanning a region and returning a prediction."""
        # synthesize realistic feature values based on lat/lon heuristics
        features = self._synthesize_features(lat, lon)
        prediction = self.predictor.predict(features)

        sat = random.choice(self.satellites)

        return {
            "satellite_id": sat["id"],
            "region":       {"lat": round(lat, 4), "lon": round(lon, 4)},
            "features":     {k: round(v, 4) for k, v in features.items()},
            "prediction":   prediction,
        }

    def _synthesize_features(self, lat, lon):
        """Generate plausible feature values for a given lat/lon."""
        # tropics → higher LST, humidity; polar → lower
        abs_lat = abs(lat)
        base_lst = 35 - abs_lat * 0.3
        base_hum = 80 - abs_lat * 0.4

        return {
            "ndvi":       round(random.gauss(0.3, 0.25), 4),
            "ndwi":       round(random.gauss(0.1, 0.3), 4),
            "lst":        round(random.gauss(base_lst, 8), 2),
            "swir":       round(abs(random.gauss(0.2, 0.15)), 4),
            "nir":        round(abs(random.gauss(0.3, 0.15)), 4),
            "precip":     round(random.gauss(20, 60), 2),
            "wind_speed": round(abs(random.gauss(30, 40)), 2),
            "humidity":   round(min(100, max(10, random.gauss(base_hum, 20))), 2),
            "soil_moist": round(min(100, max(0, random.gauss(50, 25))), 2),
            "sst_anom":   round(random.gauss(0, 1.5), 4),
            "elevation":  round(abs(random.gauss(300, 600)), 1),
            "slope":      round(abs(random.gauss(10, 12)), 2),
            "seismic_v":  round(random.gauss(0, 1), 4),
            "cloud_cov":  round(min(100, max(0, random.gauss(50, 30))), 2),
        }

    def run_global_scan(self, n_regions=20) -> list:
        """Scan n random global regions and return all predictions."""
        results = []
        for _ in range(n_regions):
            lat = random.uniform(-70, 70)
            lon = random.uniform(-180, 180)
            result = self.scan_region(lat, lon)
            results.append(result)
        return results


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  AI-Driven Satellite Swarm — Disaster Predictor")
    print("=" * 60)

    swarm = SatelliteSwarm(n_satellites=6)

    print("\n🛰️  Running global scan (20 regions)...\n")
    scan_results = swarm.run_global_scan(n_regions=20)

    alerts = [r for r in scan_results if r["prediction"]["prediction"] != "No Threat"]
    print(f"🚨 Active alerts: {len(alerts)} / {len(scan_results)} regions scanned\n")

    for alert in alerts[:5]:
        p = alert["prediction"]
        print(f"  [{p['severity'].upper():8}] {p['prediction']:12}  "
              f"Conf: {p['confidence']:5.1f}%  "
              f"Loc: ({alert['region']['lat']:.2f}, {alert['region']['lon']:.2f})  "
              f"by {alert['satellite_id']}")

    # save a sample output
    os.makedirs("data", exist_ok=True)
    with open("data/sample_scan.json", "w") as f:
        json.dump(scan_results[:5], f, indent=2)
    print("\n✅ Sample scan saved to data/sample_scan.json")
