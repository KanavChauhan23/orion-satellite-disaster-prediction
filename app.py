"""
AI-Driven Satellite Swarm — Backend API
========================================
Flask REST API that connects the AI disaster prediction model
to the frontend dashboard.

Endpoints:
  GET  /api/status          → System health + satellite status
  GET  /api/satellites      → Live satellite positions
  POST /api/predict         → Predict disaster for given features
  GET  /api/scan            → Trigger a global scan
  GET  /api/alerts          → Recent alert history
  GET  /api/region/:lat/:lon → Scan a specific region
  GET  /api/stats           → Prediction statistics
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import math
import json
import time
from datetime import datetime, timedelta
import sys
import os

# add parent path so we can import the model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # allow frontend to call the API

# ── Lazy-load the model (expensive to import TF/sklearn at startup) ──────────
_swarm = None

def get_swarm():
    global _swarm
    if _swarm is None:
        from models.disaster_predictor import SatelliteSwarm
        print("⏳ Initialising satellite swarm + training model...")
        _swarm = SatelliteSwarm(n_satellites=6)
        print("✅ Swarm ready.")
    return _swarm


# ── In-memory alert store ────────────────────────────────────────────────────
_alert_history = []
_MAX_ALERTS = 100

def _store_alert(scan_result):
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


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/status")
def status():
    swarm = get_swarm()
    return jsonify({
        "status":          "operational",
        "satellites":      len(swarm.satellites),
        "active":          sum(1 for s in swarm.satellites if s["status"] == "active"),
        "model_trained":   swarm.predictor.trained,
        "uptime_hours":    round(random.uniform(120, 9999), 1),
        "server_time":     datetime.utcnow().isoformat(),
        "coverage_pct":    round(random.uniform(87, 99), 1),
    })


@app.route("/api/satellites")
def satellites():
    swarm = get_swarm()
    # simulate orbital movement
    sats = []
    for s in swarm.satellites:
        t = time.time()
        orbital_period = 92 * 60  # ~92 min for LEO
        angle = (t % orbital_period) / orbital_period * 2 * math.pi
        lat_offset = s["inclination"] * math.sin(angle + hash(s["id"]) % 100)
        lon_offset = (t / 60) * (360 / orbital_period * 60)

        sats.append({
            **s,
            "lat": round((s["lat"] + lat_offset) % 180 - 90, 4),
            "lon": round((s["lon"] + lon_offset) % 360 - 180, 4),
            "signal_strength": round(random.uniform(88, 100), 1),
            "battery_pct":     round(random.uniform(70, 100), 1),
        })
    return jsonify(sats)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["ndvi", "ndwi", "lst", "swir", "nir", "precip",
                "wind_speed", "humidity", "soil_moist", "sst_anom",
                "elevation", "slope", "seismic_v", "cloud_cov"]

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        swarm = get_swarm()
        result = swarm.predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/scan")
def scan():
    n = int(request.args.get("n", 12))
    swarm = get_swarm()
    results = swarm.run_global_scan(n_regions=n)
    for r in results:
        _store_alert(r)
    return jsonify(results)


@app.route("/api/region/<float:lat>/<float:lon>")
def scan_region(lat, lon):
    swarm = get_swarm()
    result = swarm.scan_region(lat, lon)
    _store_alert(result)
    return jsonify(result)


@app.route("/api/alerts")
def alerts():
    limit = int(request.args.get("limit", 20))
    severity = request.args.get("severity")   # optional filter
    dtype    = request.args.get("type")       # optional filter

    filtered = list(reversed(_alert_history))  # newest first
    if severity:
        filtered = [a for a in filtered if a["severity"] == severity]
    if dtype:
        filtered = [a for a in filtered if a["type"].lower() == dtype.lower()]

    return jsonify(filtered[:limit])


@app.route("/api/stats")
def stats():
    total = len(_alert_history)
    by_type = {}
    by_severity = {}
    for a in _alert_history:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
        by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1

    return jsonify({
        "total_alerts":      total,
        "by_disaster_type":  by_type,
        "by_severity":       by_severity,
        "avg_confidence":    round(
            sum(a["confidence"] for a in _alert_history) / max(total, 1), 2
        ),
    })


@app.route("/api/history")
def history():
    """Return 7-day simulated alert history for charts."""
    rows = []
    now = datetime.utcnow()
    for day_offset in range(7):
        d = now - timedelta(days=6 - day_offset)
        rows.append({
            "date":      d.strftime("%b %d"),
            "floods":    random.randint(2, 18),
            "wildfires": random.randint(1, 12),
            "cyclones":  random.randint(0, 6),
            "earthquakes": random.randint(0, 4),
            "droughts":  random.randint(1, 8),
        })
    return jsonify(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛰️  AI Satellite Swarm API starting...")
    print("   Dashboard: http://localhost:5000")
    print("   API base:  http://localhost:5000/api/")
    # trigger model training at startup
    get_swarm()
    app.run(debug=True, port=5000)
