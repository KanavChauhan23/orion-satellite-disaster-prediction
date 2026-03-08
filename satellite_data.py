"""
Satellite Data Simulator
========================
Simulates real satellite data from NASA / Sentinel / Google Earth Engine APIs.
In a production system, these functions would make real API calls.
For the prototype, they generate realistic synthetic data.
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

# ── Known disaster-prone regions ─────────────────────────────────────────────
RISK_ZONES = {
    "Bangladesh Delta":       {"lat": 23.5, "lon": 90.0,  "risk": "flood"},
    "Amazon Basin":           {"lat": -3.0, "lon": -60.0, "risk": "drought"},
    "California":             {"lat": 37.0, "lon": -120.0,"risk": "wildfire"},
    "Philippines":            {"lat": 12.0, "lon": 122.0, "risk": "cyclone"},
    "Nepal Himalayas":        {"lat": 28.0, "lon": 84.0,  "risk": "earthquake"},
    "Sahel Region":           {"lat": 14.0, "lon": 5.0,   "risk": "drought"},
    "Indonesia":              {"lat": -3.0, "lon": 115.0, "risk": "wildfire"},
    "Pakistan Indus Valley":  {"lat": 27.0, "lon": 68.0,  "risk": "flood"},
    "Japan Pacific Coast":    {"lat": 36.0, "lon": 141.0, "risk": "earthquake"},
    "Caribbean":              {"lat": 18.0, "lon": -70.0, "risk": "cyclone"},
    "Australia Outback":      {"lat":-25.0, "lon": 134.0, "risk": "wildfire"},
    "Mozambique Coast":       {"lat":-18.0, "lon": 35.0,  "risk": "cyclone"},
}


@dataclass
class SatelliteReading:
    """Represents a single satellite sensor reading for a location."""
    satellite_id:  str
    timestamp:     str
    lat:           float
    lon:           float
    region_name:   Optional[str]

    # Optical bands
    band_red:   float   # Red reflectance (0-1)
    band_nir:   float   # Near-infrared (0-1)
    band_swir:  float   # Short-wave infrared (0-1)

    # Derived indices
    ndvi:       float   # Vegetation index
    ndwi:       float   # Water index
    evi:        float   # Enhanced vegetation index

    # Thermal
    lst:        float   # Land surface temperature (°C)

    # Atmospheric / weather
    precip:     float   # Precipitation anomaly (mm/day)
    wind_speed: float   # km/h
    humidity:   float   # %
    cloud_cov:  float   # %

    # Surface
    soil_moist: float   # %
    sst_anom:   float   # Sea surface temp anomaly (°C)

    # Terrain
    elevation:  float   # metres
    slope:      float   # degrees

    # Geophysical
    seismic_v:  float   # velocity anomaly


class SatelliteDataSimulator:
    """
    Simulates the data pipeline from real satellite APIs.

    Real equivalents:
    - NASA FIRMS  → fire / hotspot data
    - NASA GRACE  → soil moisture / groundwater
    - Sentinel-1  → SAR (floods, earthquakes)
    - Sentinel-2  → multispectral (vegetation, water)
    - Sentinel-3  → sea surface temperature, fire
    - MODIS       → LST, vegetation
    - NOAA GOES   → weather / hurricanes
    """

    SAT_IDS = [f"SAT-{1001+i}" for i in range(6)]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _base_features(self, lat: float, lon: float) -> dict:
        """Generate base realistic feature values for a location."""
        abs_lat = abs(lat)

        # Temperature varies by latitude
        base_lst = 35 - abs_lat * 0.35 + self.rng.normal(0, 5)
        base_hum = 75 - abs_lat * 0.3 + self.rng.normal(0, 12)

        band_red  = float(np.clip(self.rng.normal(0.12, 0.06), 0, 0.5))
        band_nir  = float(np.clip(self.rng.normal(0.35, 0.15), 0, 0.7))
        band_swir = float(np.clip(self.rng.normal(0.18, 0.08), 0, 0.6))

        ndvi = (band_nir - band_red) / (band_nir + band_red + 1e-8)
        ndwi = (band_nir - band_swir) / (band_nir + band_swir + 1e-8)
        evi  = 2.5 * (band_nir - band_red) / (band_nir + 6*band_red - 7.5*0.03 + 1 + 1e-8)

        return {
            "band_red":   round(band_red, 4),
            "band_nir":   round(band_nir, 4),
            "band_swir":  round(band_swir, 4),
            "ndvi":       round(float(np.clip(ndvi, -1, 1)), 4),
            "ndwi":       round(float(np.clip(ndwi, -1, 1)), 4),
            "evi":        round(float(np.clip(evi,  -1, 1)), 4),
            "lst":        round(float(np.clip(base_lst, -20, 60)), 2),
            "precip":     round(float(self.rng.normal(15, 50)), 2),
            "wind_speed": round(float(np.clip(self.rng.normal(25, 35), 0, 200)), 2),
            "humidity":   round(float(np.clip(base_hum, 10, 100)), 2),
            "cloud_cov":  round(float(np.clip(self.rng.normal(45, 30), 0, 100)), 2),
            "soil_moist": round(float(np.clip(self.rng.normal(45, 22), 0, 100)), 2),
            "sst_anom":   round(float(self.rng.normal(0, 1.2)), 4),
            "elevation":  round(float(np.clip(self.rng.exponential(350), 0, 5000)), 1),
            "slope":      round(float(np.clip(self.rng.exponential(10), 0, 45)), 2),
            "seismic_v":  round(float(self.rng.normal(0, 0.7)), 4),
        }

    def _apply_disaster_signature(self, features: dict, disaster_type: str) -> dict:
        """Modify features to match known disaster signatures."""
        f = features.copy()

        if disaster_type == "flood":
            f["ndwi"]       = round(np.clip(f["ndwi"] + 0.45, -1, 1), 4)
            f["precip"]     = round(max(f["precip"] + 100, 90), 2)
            f["soil_moist"] = round(np.clip(f["soil_moist"] + 30, 0, 100), 2)
            f["elevation"]  = round(np.clip(f["elevation"], 0, 300), 1)
            f["cloud_cov"]  = round(np.clip(f["cloud_cov"] + 30, 0, 100), 2)

        elif disaster_type == "wildfire":
            f["lst"]        = round(np.clip(f["lst"] + 18, 0, 60), 2)
            f["ndvi"]       = round(np.clip(f["ndvi"] - 0.3, -1, 1), 4)
            f["humidity"]   = round(np.clip(f["humidity"] - 35, 10, 100), 2)
            f["swir"]       = round(np.clip(f["band_swir"] + 0.25, 0, 0.6), 4)
            f["soil_moist"] = round(np.clip(f["soil_moist"] - 30, 0, 100), 2)

        elif disaster_type == "earthquake":
            f["seismic_v"]  = round(np.clip(f["seismic_v"] + 1.7 * np.sign(f["seismic_v"] or 1), -2, 2), 4)
            f["slope"]      = round(np.clip(f["slope"] + 20, 0, 45), 2)
            f["elevation"]  = round(np.clip(f["elevation"] + 500, 0, 5000), 1)

        elif disaster_type == "cyclone":
            f["wind_speed"] = round(np.clip(f["wind_speed"] + 110, 0, 200), 2)
            f["sst_anom"]   = round(np.clip(f["sst_anom"] + 2.0, -3, 5), 4)
            f["cloud_cov"]  = round(np.clip(f["cloud_cov"] + 35, 0, 100), 2)
            f["precip"]     = round(f["precip"] + 60, 2)
            f["humidity"]   = round(np.clip(f["humidity"] + 20, 0, 100), 2)

        elif disaster_type == "drought":
            f["ndvi"]       = round(np.clip(f["ndvi"] - 0.25, -1, 1), 4)
            f["soil_moist"] = round(np.clip(f["soil_moist"] - 35, 0, 100), 2)
            f["lst"]        = round(np.clip(f["lst"] + 10, 0, 60), 2)
            f["precip"]     = round(min(f["precip"] - 45, -20), 2)
            f["humidity"]   = round(np.clip(f["humidity"] - 30, 10, 100), 2)

        return f

    def read_location(
        self,
        lat: float,
        lon: float,
        region_name: Optional[str] = None,
        force_disaster: Optional[str] = None
    ) -> SatelliteReading:
        """Simulate a single satellite reading for a lat/lon location."""
        features = self._base_features(lat, lon)
        if force_disaster:
            features = self._apply_disaster_signature(features, force_disaster)

        return SatelliteReading(
            satellite_id = random.choice(self.SAT_IDS),
            timestamp    = datetime.utcnow().isoformat() + "Z",
            lat          = round(lat, 4),
            lon          = round(lon, 4),
            region_name  = region_name,
            **features,
        )

    def scan_risk_zones(self) -> List[SatelliteReading]:
        """Scan all known high-risk zones."""
        readings = []
        for name, info in RISK_ZONES.items():
            # add small jitter so readings aren't always exactly at zone centre
            lat = info["lat"] + self.rng.normal(0, 0.5)
            lon = info["lon"] + self.rng.normal(0, 0.5)
            # 60% chance the risk zone is actually showing its disaster signature
            disaster = info["risk"] if self.rng.random() < 0.6 else None
            r = self.read_location(lat, lon, region_name=name, force_disaster=disaster)
            readings.append(r)
        return readings

    def generate_time_series(
        self,
        lat: float,
        lon: float,
        days: int = 30,
        disaster_onset_day: int = 20
    ) -> pd.DataFrame:
        """
        Generate a 30-day time series for one location,
        with a disaster building up after `disaster_onset_day`.
        """
        rows = []
        disaster = random.choice(["flood", "wildfire", "cyclone"])
        for d in range(days):
            ts = datetime.utcnow() - timedelta(days=days - d)
            features = self._base_features(lat, lon)
            if d >= disaster_onset_day:
                intensity = (d - disaster_onset_day) / (days - disaster_onset_day)
                partial = {k: v for k, v in self._apply_disaster_signature(features, disaster).items()}
                # blend: increase intensity gradually
                for k in features:
                    features[k] = features[k] + (partial[k] - features[k]) * intensity
            reading = {
                "date":    ts.strftime("%Y-%m-%d"),
                "lat":     round(lat, 4),
                "lon":     round(lon, 4),
                "disaster": disaster if d >= disaster_onset_day else "none",
                **features
            }
            rows.append(reading)
        return pd.DataFrame(rows)

    def export_json(self, readings: List[SatelliteReading], path: str):
        with open(path, "w") as f:
            json.dump([asdict(r) for r in readings], f, indent=2)
        print(f"📁 Exported {len(readings)} readings → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = SatelliteDataSimulator()

    print("🛰  Scanning known high-risk zones...")
    readings = sim.scan_risk_zones()
    sim.export_json(readings, "data/risk_zone_readings.json")

    print("\n📈 Generating 30-day time series (Bangladesh Delta)...")
    zone = RISK_ZONES["Bangladesh Delta"]
    ts_df = sim.generate_time_series(zone["lat"], zone["lon"], days=30)
    ts_df.to_csv("data/time_series_bangladesh.csv", index=False)
    print(f"   Saved to data/time_series_bangladesh.csv ({len(ts_df)} rows)")

    print("\nSample reading:")
    r = readings[0]
    print(f"  Location: {r.region_name} ({r.lat}, {r.lon})")
    print(f"  NDVI={r.ndvi}  NDWI={r.ndwi}  LST={r.lst}°C")
    print(f"  Wind={r.wind_speed}km/h  Humidity={r.humidity}%")
    print(f"  SeismicV={r.seismic_v}  Slope={r.slope}°")
