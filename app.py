"""
AI-Driven Satellite Swarm — Backend API
========================================
Flask REST API connecting the AI disaster prediction model
to the frontend dashboard.

Endpoints:
  GET  /                      → Project info
  GET  /api/status            → System health
  GET  /api/satellites        → Live satellite positions
  POST /api/predict           → Predict disaster for given features
  GET  /api/scan              → Trigger a global scan
  GET  /api/alerts            → Recent alert history
  GET  /api/region/<lat>/<lon>→ Scan a specific region
  GET  /api/stats             → Prediction statistics
  GET  /api/history           → 7-day alert history
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import math
import json
import time
import os
from datetime import datetime, timedelta
import sys

# add models folder to path so we can import disaster_predictor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

app = Flask(__name__)
CORS(app)  # allow frontend (Vercel) to call this API

# ── Lazy-load the AI model ─────────────────────────────────────────────────
_swarm = None

def get_swarm():
    global _swarm
    if _swarm is None:
        try:
            from disaster_predictor import SatelliteSwarm
            print("⏳ Initialising satellite swarm + training AI model...")
            _swarm = SatelliteSwarm(n_satellites=6)
            print("✅ Swarm ready.")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")
            _swarm = None
    return _swarm


# ── In-memory alert store ──────────────────────────────────────────────────
_alert_history = []
_MAX_ALERTS    = 100

def _store_alert(scan_result):
    """Save non-trivial predictions to in-memory history."""
    p = scan_result["prediction"]
    if p["prediction"] != "No Threat":
        _alert_history.append({
            "id":           len(_alert_history) + 1,
            "satellite_id": scan_result["satellite_id"],
            "type":         p["prediction"],
            "severity":     p["severity"],
            "confidence":   p["confidence"],
            "lat":          scan_result["region"]["lat"],
            "lon":          scan_result["region"]["lon"],
            "timestamp":    p["timestamp"],
        })
        if len(_alert_history) > _MAX_ALERTS:
            _alert_history.pop(0)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Root route — confirms API is alive."""
    return jsonify({
        "project":     "ORION — AI-Driven Satellite Swarm",
        "version":     "1.0.0",
        "status":      "running",
        "description": "Autonomous Disaster Prediction System",
        "endpoints": {
            "status":     "/api/status",
            "satellites": "/api/satellites",
            "predict":    "/api/predict  [POST]",
            "scan":       "/api/scan",
            "alerts":     "/api/alerts",
            "stats":      "/api/stats",
            "history":    "/api/history",
        }
    })


@app.route("/api/status")
def status():
    """System health check."""
    swarm = get_swarm()
    return jsonify({
        "status":        "operational",
        "satellites":    6,
        "active":        6,
        "model_trained": swarm is not None,
        "uptime_hours":  round(random.uniform(120, 9999), 1),
        "server_time":   datetime.utcnow().isoformat(),
        "coverage_pct":  round(random.uniform(87, 99), 1),
    })


@app.route("/api/satellites")
def satellites():
    """Return live simulated satellite positions."""
    sats = []
    inclinations = [53, 97.6, 45, 70, 53, 97.6]
    altitudes    = [450, 620, 510, 780, 430, 680]
    orbit_types  = ["LEO", "SSO", "LEO", "LEO", "LEO", "SSO"]

    for i in range(6):
        t             = time.time()
        orbital_period = 92 * 60
        angle         = (t % orbital_period) / orbital_period * 2 * math.pi
        lat_offset    = inclinations[i] * math.sin(angle + i * 1.05)
        lon_offset    = (t / 60) * (360 / (orbital_period / 60))

        sats.append({
            "id":              f"SAT-{1001 + i}",
            "altitude_km":     altitudes[i],
            "orbit_type":      orbit_types[i],
            "inclination":     inclinations[i],
            "sensors":         ["optical", "SAR", "thermal", "multispectral"],
            "status":          "active",
            "lat":             round((lat_offset) % 180 - 90, 4),
            "lon":             round((lon_offset + i * 60) % 360 - 180, 4),
            "coverage_km":     random.randint(200, 600),
            "signal_strength": round(random.uniform(88, 100), 1),
            "battery_pct":     round(random.uniform(70, 100), 1),
        })
    return jsonify(sats)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict disaster risk from satellite feature inputs.

    Required JSON body fields:
      ndvi, ndwi, lst, swir, nir, precip, wind_speed,
      humidity, soil_moist, sst_anom, elevation, slope,
      seismic_v, cloud_cov
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = [
        "ndvi", "ndwi", "lst", "swir", "nir", "precip",
        "wind_speed", "humidity", "soil_moist", "sst_anom",
        "elevation", "slope", "seismic_v", "cloud_cov"
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    swarm = get_swarm()
    if swarm is None:
        return jsonify({"error": "AI model not available. Try again in a few seconds."}), 503

    try:
        result = swarm.predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scan")
def scan():
    """Trigger a global scan of n random regions."""
    n = int(request.args.get("n", 12))
    n = min(n, 50)  # cap at 50 to prevent abuse

    swarm = get_swarm()
    if swarm is None:
        return jsonify({"error": "AI model not available."}), 503

    try:
        results = swarm.run_global_scan(n_regions=n)
        for r in results:
            _store_alert(r)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/region/<float:lat>/<float:lon>")
def scan_region(lat, lon):
    """Scan a specific lat/lon region."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "Invalid coordinates"}), 400

    swarm = get_swarm()
    if swarm is None:
        return jsonify({"error": "AI model not available."}), 503

    try:
        result = swarm.scan_region(lat, lon)
        _store_alert(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alerts")
def alerts():
    """Return recent alert history with optional filters."""
    limit    = int(request.args.get("limit", 20))
    severity = request.args.get("severity")
    dtype    = request.args.get("type")

    filtered = list(reversed(_alert_history))
    if severity:
        filtered = [a for a in filtered if a["severity"] == severity]
    if dtype:
        filtered = [a for a in filtered if a["type"].lower() == dtype.lower()]

    return jsonify(filtered[:limit])


@app.route("/api/stats")
def stats():
    """Return prediction statistics."""
    total    = len(_alert_history)
    by_type  = {}
    by_sev   = {}

    for a in _alert_history:
        by_type[a["type"]]    = by_type.get(a["type"], 0) + 1
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1

    return jsonify({
        "total_alerts":     total,
        "by_disaster_type": by_type,
        "by_severity":      by_sev,
        "avg_confidence":   round(
            sum(a["confidence"] for a in _alert_history) / max(total, 1), 2
        ),
    })


@app.route("/api/history")
def history():
    """Return 7-day simulated alert history for charts."""
    rows = []
    now  = datetime.utcnow()
    for day_offset in range(7):
        d = now - timedelta(days=6 - day_offset)
        rows.append({
            "date":        d.strftime("%b %d"),
            "floods":      random.randint(2, 18),
            "wildfires":   random.randint(1, 12),
            "cyclones":    random.randint(0, 6),
            "earthquakes": random.randint(0, 4),
            "droughts":    random.randint(1, 8),
        })
    return jsonify(rows)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛰️  ORION AI Satellite Swarm API starting...")
    print("   Local URL: http://localhost:5000")
    print("   API docs:  http://localhost:5000/")
    # Read PORT from environment (Render sets this automatically)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
