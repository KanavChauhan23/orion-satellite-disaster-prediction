"""
Satellite Data Simulator
========================
Simulates real satellite data from NASA / Sentinel / Google Earth Engine APIs.
In production, these functions make real API calls.
For the prototype they generate realistic synthetic data.

Real data sources this module simulates:
  NASA Earthdata  → MODIS LST, GRACE soil moisture, FIRMS fire hotspots
  Sentinel-1      → SAR (floods, earthquakes surface deformation)
  Sentinel-2      → Multispectral (NDVI, NDWI, SWIR)
  Sentinel-3      → Sea surface temperature, fire radiative power
  NOAA GOES       → Weather, hurricane tracking
  USGS            → Seismic network, elevation (SRTM DEM)
  NASA GPM        → Global precipitation measurement
"""

import numpy as np
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Optional


# ── Known disaster-prone regions ─────────────────────────────────────────────
RISK_ZONES = {
    "Bangladesh Delta":       {"lat":  23.5, "lon":  90.0,  "primary_risk": "flood"},
    "Amazon Basin":           {"lat":  -3.0, "lon": -60.0,  "primary_risk": "drought"},
    "California Foothills":   {"lat":  37.0, "lon":-120.0,  "primary_risk": "wildfire"},
    "Philippines":            {"lat":  12.0, "lon": 122.0,  "primary_risk": "cyclone"},
    "Nepal Himalayas":        {"lat":  28.0, "lon":  84.0,  "primary_risk": "earthquake"},
    "Sahel Region":           {"lat":  14.0, "lon":   5.0,  "primary_risk": "drought"},
    "Borneo / Indonesia":     {"lat":  -3.0, "lon": 115.0,  "primary_risk": "wildfire"},
    "Pakistan Indus Valley":  {"lat":  27.0, "lon":  68.0,  "primary_risk": "flood"},
    "Japan Pacific Coast":    {"lat":  36.0, "lon": 141.0,  "primary_risk": "earthquake"},
    "Caribbean Basin":        {"lat":  18.0, "lon": -70.0,  "primary_risk": "cyclone"},
    "Australia Outback":      {"lat": -25.0, "lon": 134.0,  "primary_risk": "wildfire"},
    "Mozambique Coast":       {"lat": -18.0, "lon":  35.0,  "primary_risk": "cyclone"},
    "Horn of Africa":         {"lat":   8.0, "lon":  43.0,  "primary_risk": "drought"},
    "Central America":        {"lat":  15.0, "lon": -87.0,  "primary_risk": "flood"},
    "Turkey Anatolia":        {"lat":  39.0, "lon":  35.0,  "primary_risk": "earthquake"},
}


@dataclass
class SatelliteReading:
    """Represents one satellite sensor reading snapshot for a location."""

    satellite_id:  str
    timestamp:     str
    lat:           float
    lon:           float
    region_name:   Optional[str]

    # ── Optical bands (Sentinel-2) ──────────────
    band_red:   float    # Red reflectance        0–1
    band_nir:   float    # Near-Infrared          0–1
    band_swir:  float    # Short-Wave Infrared    0–1

    # ── Derived spectral indices ─────────────────
    ndvi:       float    # Vegetation index       -1–1
    ndwi:       float    # Water index            -1–1
    evi:        float    # Enhanced vegetation    -1–1

    # ── Thermal (MODIS / Landsat) ────────────────
    lst:        float    # Land surface temp °C

    # ── Atmospheric / weather ────────────────────
    precip:     float    # Precipitation anomaly mm/day
    wind_speed: float    # Wind speed km/h
    humidity:   float    # Relative humidity %
    cloud_cov:  float    # Cloud coverage %

    # ── Surface conditions ───────────────────────
    soil_moist: float    # Soil moisture %
    sst_anom:   float    # SST anomaly °C

    # ── Terrain (SRTM DEM) ───────────────────────
    elevation:  float    # Height m
    slope:      float    # Slope degrees

    # ── Geophysical (USGS seismic) ───────────────
    seismic_v:  float    # Velocity anomaly


class SatelliteDataSimulator:
    """
    Simulates the full satellite data pipeline.

    Usage:
        sim = SatelliteDataSimulator()
        readings = sim.scan_risk_zones()
        sim.export_json(readings, "data/readings.json")
    """

    SAT_IDS = [f"SAT-{1001 + i}" for i in range(6)]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    # ── Feature synthesis ─────────────────────────────────────
    def _base_features(self, lat: float, lon: float) -> dict:
        """Generate realistic baseline features for any lat/lon."""
        abs_lat   = abs(lat)
        base_lst  = 35 - abs_lat * 0.35 + float(self.rng.normal(0, 5))
        base_hum  = 75 - abs_lat * 0.30 + float(self.rng.normal(0, 12))

        band_red  = float(np.clip(self.rng.normal(0.12, 0.06), 0.0, 0.50))
        band_nir  = float(np.clip(self.rng.normal(0.35, 0.15), 0.0, 0.70))
        band_swir = float(np.clip(self.rng.normal(0.18, 0.08), 0.0, 0.60))

        ndvi = (band_nir - band_red)  / (band_nir + band_red  + 1e-8)
        ndwi = (band_nir - band_swir) / (band_nir + band_swir + 1e-8)
        evi  = 2.5 * (band_nir - band_red) / (
            band_nir + 6 * band_red - 7.5 * 0.03 + 1 + 1e-8
        )

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
        """Modify feature values to match the geophysical signature of a disaster."""
        f = features.copy()

        if disaster_type == "flood":
            f["ndwi"]       = round(float(np.clip(f["ndwi"] + 0.50, -1, 1)), 4)
            f["precip"]     = round(max(f["precip"] + 110, 90), 2)
            f["soil_moist"] = round(float(np.clip(f["soil_moist"] + 35, 0, 100)), 2)
            f["elevation"]  = round(float(np.clip(f["elevation"],   0, 300)), 1)
            f["cloud_cov"]  = round(float(np.clip(f["cloud_cov"] + 30, 0, 100)), 2)

        elif disaster_type == "wildfire":
            f["lst"]        = round(float(np.clip(f["lst"] + 18, 0, 60)), 2)
            f["ndvi"]       = round(float(np.clip(f["ndvi"] - 0.30, -1, 1)), 4)
            f["humidity"]   = round(float(np.clip(f["humidity"] - 35, 10, 100)), 2)
            f["band_swir"]  = round(float(np.clip(f["band_swir"] + 0.25, 0, 0.6)), 4)
            f["soil_moist"] = round(float(np.clip(f["soil_moist"] - 30, 0, 100)), 2)

        elif disaster_type == "earthquake":
            sign           = 1 if f["seismic_v"] >= 0 else -1
            f["seismic_v"] = round(float(np.clip(f["seismic_v"] + sign * 1.7, -2, 2)), 4)
            f["slope"]     = round(float(np.clip(f["slope"] + 22, 0, 45)), 2)
            f["elevation"] = round(float(np.clip(f["elevation"] + 600, 0, 5000)), 1)

        elif disaster_type == "cyclone":
            f["wind_speed"] = round(float(np.clip(f["wind_speed"] + 115, 0, 200)), 2)
            f["sst_anom"]   = round(float(np.clip(f["sst_anom"]   + 2.2, -3, 5)), 4)
            f["cloud_cov"]  = round(float(np.clip(f["cloud_cov"]  + 40,  0, 100)), 2)
            f["precip"]     = round(f["precip"] + 65, 2)
            f["humidity"]   = round(float(np.clip(f["humidity"]   + 20,  0, 100)), 2)

        elif disaster_type == "drought":
            f["ndvi"]       = round(float(np.clip(f["ndvi"]       - 0.28, -1, 1)), 4)
            f["soil_moist"] = round(float(np.clip(f["soil_moist"] - 38, 0, 100)), 2)
            f["lst"]        = round(float(np.clip(f["lst"]        + 10,  0, 60)), 2)
            f["precip"]     = round(min(f["precip"] - 50, -20), 2)
            f["humidity"]   = round(float(np.clip(f["humidity"]   - 30, 10, 100)), 2)

        return f

    # ── Public API ────────────────────────────────────────────
    def read_location(
        self,
        lat:           float,
        lon:           float,
        region_name:   Optional[str] = None,
        force_disaster: Optional[str] = None,
    ) -> SatelliteReading:
        """Simulate a single satellite reading for a lat/lon."""
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
        """Scan all 15 known high-risk zones."""
        readings = []
        for name, info in RISK_ZONES.items():
            lat      = info["lat"] + float(self.rng.normal(0, 0.5))
            lon      = info["lon"] + float(self.rng.normal(0, 0.5))
            # 60 % chance the zone is actively showing its disaster signature
            disaster = info["primary_risk"] if self.rng.random() < 0.60 else None
            r = self.read_location(lat, lon, region_name=name, force_disaster=disaster)
            readings.append(r)
        return readings

    def generate_time_series(
        self,
        lat:               float,
        lon:               float,
        days:              int = 30,
        disaster_onset_day: int = 20,
        disaster_type:     Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a daily time series for one location over `days` days,
        with a disaster signature gradually building from `disaster_onset_day`.
        """
        rows     = []
        disaster = disaster_type or random.choice(["flood", "wildfire", "cyclone"])

        for d in range(days):
            ts       = datetime.utcnow() - timedelta(days=days - d)
            features = self._base_features(lat, lon)

            if d >= disaster_onset_day:
                intensity  = (d - disaster_onset_day) / max(days - disaster_onset_day, 1)
                disaster_f = self._apply_disaster_signature(features.copy(), disaster)
                for k in features:
                    features[k] = features[k] + (disaster_f[k] - features[k]) * intensity

            row = {
                "date":     ts.strftime("%Y-%m-%d"),
                "lat":      round(lat, 4),
                "lon":      round(lon, 4),
                "disaster": disaster if d >= disaster_onset_day else "none",
                **features,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def export_json(self, readings: List[SatelliteReading], path: str):
        """Export a list of SatelliteReading objects to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump([asdict(r) for r in readings], fh, indent=2)
        print(f"📁 Exported {len(readings)} readings → {path}")


# need os for makedirs
import os


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ORION — Satellite Data Simulator")
    print("=" * 60)

    sim = SatelliteDataSimulator()

    print("\n🛰️  Scanning 15 high-risk zones …")
    readings = sim.scan_risk_zones()
    os.makedirs("data", exist_ok=True)
    sim.export_json(readings, "data/risk_zone_readings.json")

    print("\n📈 Generating 30-day time series — Bangladesh Delta …")
    zone = RISK_ZONES["Bangladesh Delta"]
    ts_df = sim.generate_time_series(
        zone["lat"], zone["lon"],
        days=30, disaster_onset_day=18, disaster_type="flood"
    )
    ts_df.to_csv("data/time_series_bangladesh.csv", index=False)
    print(f"   Saved → data/time_series_bangladesh.csv  ({len(ts_df)} rows)")

    print("\nSample reading:")
    r = readings[0]
    print(f"  Region   : {r.region_name}")
    print(f"  Location : ({r.lat}, {r.lon})")
    print(f"  NDVI={r.ndvi}  NDWI={r.ndwi}  LST={r.lst}°C")
    print(f"  Wind={r.wind_speed} km/h  Humidity={r.humidity}%")
    print(f"  SeismicV={r.seismic_v}  Slope={r.slope}°")
    print(f"  Satellite: {r.satellite_id}")
