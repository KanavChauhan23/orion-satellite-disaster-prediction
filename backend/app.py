"""
ORION — Backend API (Patent Edition)
======================================
Includes all 3 novel patent algorithm endpoints:

  POST /api/cascade          → CLAIM 1: Cascade disaster prediction
  POST /api/tewi             → CLAIM 2: Temporal Early Warning Index
  POST /api/dspr             → CLAIM 3: Dynamic Swarm Priority Reassignment
  POST /api/patent/analysis  → All 3 claims in one unified call
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random, math, time, os, sys, json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

app  = Flask(__name__)
CORS(app)

# ── Lazy-load models ──────────────────────────────────────────────
_swarm         = None
_patent_engine = None

def get_swarm():
    global _swarm
    if _swarm is None:
        try:
            from disaster_predictor import SatelliteSwarm
            print("⏳ Training AI model...")
            _swarm = SatelliteSwarm(n_satellites=6)
            print("✅ Model ready.")
        except Exception as e:
            print(f"⚠️  Model load failed: {e}")
    return _swarm

def get_patent_engine():
    global _patent_engine
    if _patent_engine is None:
        try:
            from patent_algorithms import ORIONPatentEngine
            _patent_engine = ORIONPatentEngine(n_satellites=6)
            print("✅ Patent engine ready.")
        except Exception as e:
            print(f"⚠️  Patent engine load failed: {e}")
    return _patent_engine

# ── Alert store ───────────────────────────────────────────────────
_alert_history = []

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
        if len(_alert_history) > 100:
            _alert_history.pop(0)


# ════════════════════════════════════════════════════════════════
# EXISTING ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return jsonify({
        "project":     "ORION — AI-Driven Satellite Swarm",
        "version":     "2.0.0-patent",
        "status":      "running",
        "patent_claims": {
            "claim_1": "Cascade Disaster Prediction  →  POST /api/cascade",
            "claim_2": "Temporal Early Warning Index →  POST /api/tewi",
            "claim_3": "Dynamic Swarm Priority       →  POST /api/dspr",
            "unified": "All 3 claims combined        →  POST /api/patent/analysis",
        },
        "standard_endpoints": {
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
    swarm = get_swarm()
    return jsonify({
        "status":        "operational",
        "satellites":    6,
        "active":        6,
        "model_trained": swarm is not None,
        "patent_engine": get_patent_engine() is not None,
        "server_time":   datetime.utcnow().isoformat(),
        "coverage_pct":  round(random.uniform(87, 99), 1),
    })


@app.route("/api/satellites")
def satellites():
    sats = []
    inclinations = [53, 97.6, 45, 70, 53, 97.6]
    altitudes    = [450, 620, 510, 780, 430, 680]
    orbit_types  = ["LEO","SSO","LEO","LEO","LEO","SSO"]
    for i in range(6):
        t  = time.time()
        op = 92 * 60
        a  = (t % op) / op * 2 * math.pi
        sats.append({
            "id":              f"SAT-{1001+i}",
            "altitude_km":     altitudes[i],
            "orbit_type":      orbit_types[i],
            "inclination":     inclinations[i],
            "sensors":         ["optical","SAR","thermal","multispectral"],
            "status":          "active",
            "lat":             round((inclinations[i]*math.sin(a+i*1.05))%180-90, 4),
            "lon":             round((t/60*(360/(op/60))+i*60)%360-180, 4),
            "signal_strength": round(random.uniform(88,100), 1),
            "battery_pct":     round(random.uniform(70,100), 1),
        })
    return jsonify(sats)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    required = ["ndvi","ndwi","lst","swir","nir","precip","wind_speed",
                "humidity","soil_moist","sst_anom","elevation","slope",
                "seismic_v","cloud_cov"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    swarm = get_swarm()
    if not swarm:
        return jsonify({"error": "Model unavailable"}), 503
    return jsonify(swarm.predictor.predict(data))


@app.route("/api/scan")
def scan():
    n     = min(int(request.args.get("n", 12)), 50)
    swarm = get_swarm()
    if not swarm:
        return jsonify({"error": "Model unavailable"}), 503
    results = swarm.run_global_scan(n_regions=n)
    for r in results:
        _store_alert(r)
    return jsonify(results)


@app.route("/api/region/<float:lat>/<float:lon>")
def scan_region(lat, lon):
    if not (-90<=lat<=90 and -180<=lon<=180):
        return jsonify({"error": "Invalid coordinates"}), 400
    swarm = get_swarm()
    if not swarm:
        return jsonify({"error": "Model unavailable"}), 503
    result = swarm.scan_region(lat, lon)
    _store_alert(result)
    return jsonify(result)


@app.route("/api/alerts")
def alerts():
    limit    = int(request.args.get("limit", 20))
    severity = request.args.get("severity")
    dtype    = request.args.get("type")
    filtered = list(reversed(_alert_history))
    if severity: filtered = [a for a in filtered if a["severity"]==severity]
    if dtype:    filtered = [a for a in filtered if a["type"].lower()==dtype.lower()]
    return jsonify(filtered[:limit])


@app.route("/api/stats")
def stats():
    total   = len(_alert_history)
    by_type = {}
    by_sev  = {}
    for a in _alert_history:
        by_type[a["type"]]    = by_type.get(a["type"], 0) + 1
        by_sev[a["severity"]] = by_sev.get(a["severity"], 0) + 1
    return jsonify({
        "total_alerts":     total,
        "by_disaster_type": by_type,
        "by_severity":      by_sev,
        "avg_confidence":   round(sum(a["confidence"] for a in _alert_history)/max(total,1),2),
    })


@app.route("/api/history")
def history():
    rows = []
    now  = datetime.utcnow()
    for d in range(7):
        dt = now - timedelta(days=6-d)
        rows.append({
            "date":        dt.strftime("%b %d"),
            "floods":      random.randint(2,18),
            "wildfires":   random.randint(1,12),
            "cyclones":    random.randint(0,6),
            "earthquakes": random.randint(0,4),
            "droughts":    random.randint(1,8),
        })
    return jsonify(rows)


# ════════════════════════════════════════════════════════════════
# PATENT CLAIM ROUTES
# ════════════════════════════════════════════════════════════════

@app.route("/api/cascade", methods=["POST"])
def cascade():
    """
    PATENT CLAIM 1 — Cascade Disaster Prediction
    
    POST body:
      {
        "primary_disaster": "Drought",
        "confidence": 88.5,
        "features": { ndvi, ndwi, lst, ... }
      }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    primary    = data.get("primary_disaster")
    confidence = data.get("confidence", 70.0)
    features   = data.get("features", {})

    if not primary:
        return jsonify({"error": "primary_disaster is required"}), 400

    engine = get_patent_engine()
    if not engine:
        return jsonify({"error": "Patent engine unavailable"}), 503

    try:
        cascades = engine.cascade_engine.predict_cascades(
            primary, features, confidence
        )
        return jsonify({
            "claim":            "1 - Cascade Disaster Prediction",
            "primary_disaster": primary,
            "primary_confidence": confidence,
            "cascade_predictions": [
                {
                    "triggered_disaster":  c.triggered_disaster,
                    "cascade_probability": c.cascade_probability,
                    "combined_severity":   c.combined_severity,
                    "estimated_delay_hrs": c.estimated_delay_hrs,
                    "amplifier_features":  c.amplifier_features,
                }
                for c in cascades
            ],
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tewi", methods=["POST"])
def tewi():
    """
    PATENT CLAIM 2 — Temporal Early Warning Index
    
    POST body:
      {
        "time_series": [ {feature_dict_t0}, {feature_dict_t1}, ... ],
        "window_hours": 72
      }
      OR provide base features + disaster_type to auto-generate series:
      {
        "features": { ndvi, ndwi, ... },
        "disaster_type": "Wildfire",
        "auto_generate": true
      }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    engine = get_patent_engine()
    if not engine:
        return jsonify({"error": "Patent engine unavailable"}), 503

    try:
        time_series = data.get("time_series")

        # auto-generate a demonstration time series
        if not time_series and data.get("auto_generate"):
            features     = data.get("features", {})
            disaster_type= data.get("disaster_type", "Wildfire")
            time_series  = engine.tewi_engine.generate_synthetic_timeseries(
                features, disaster_type, hours=72, readings=12
            )

        if not time_series or len(time_series) < 2:
            return jsonify({"error": "time_series with >=2 readings required"}), 400

        window  = int(data.get("window_hours", 72))
        results = engine.tewi_engine.compute_tewi(time_series, window)

        return jsonify({
            "claim":        "2 - Temporal Early Warning Index (TEWI)",
            "window_hours": window,
            "readings_used":len(time_series),
            "early_warnings": [
                {
                    "disaster_type":   r.disaster_type,
                    "tewi_score":      r.tewi_score,
                    "warning_level":   r.warning_level,
                    "hours_to_onset":  r.hours_to_onset,
                    "trend_direction": r.trend_direction,
                    "key_indicators":  r.key_indicators,
                    "confidence":      r.confidence,
                }
                for r in results
            ],
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dspr", methods=["POST"])
def dspr():
    """
    PATENT CLAIM 3 — Dynamic Swarm Priority Reassignment
    
    POST body:
      {
        "candidate_regions": [ [lat, lon], [lat, lon], ... ],
        "active_alerts": [ {type, lat, lon, confidence}, ... ]
      }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    engine = get_patent_engine()
    if not engine:
        return jsonify({"error": "Patent engine unavailable"}), 503

    try:
        regions       = data.get("candidate_regions", [])
        active_alerts = data.get("active_alerts", _alert_history[-10:])

        # if no regions provided, generate 12 random global regions
        if not regions:
            regions = [
                [random.uniform(-65,65), random.uniform(-175,175)]
                for _ in range(12)
            ]

        features_map = {
            engine.dspr_engine._region_key(r[0], r[1]): {}
            for r in regions
        }

        assignments = engine.dspr_engine.assign_satellites(
            [(r[0],r[1]) for r in regions],
            features_map,
            active_alerts
        )
        coverage = engine.dspr_engine.get_coverage_report(assignments)

        return jsonify({
            "claim":           "3 - Dynamic Swarm Priority Reassignment (DSPR)",
            "n_satellites":    6,
            "n_regions_scored":len(regions),
            "assignments": [
                {
                    "satellite_id":          a.satellite_id,
                    "target_lat":            a.target_lat,
                    "target_lon":            a.target_lon,
                    "priority_score":        a.priority_score,
                    "reason":                a.reason,
                    "cascade_risk":          a.cascade_risk,
                    "population_factor":     a.population_factor,
                    "scan_time_min":         a.estimated_scan_time_min,
                }
                for a in assignments
            ],
            "coverage_report": coverage,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/patent/analysis", methods=["POST"])
def patent_analysis():
    """
    UNIFIED — All 3 patent claims in a single API call.
    
    POST body:
      {
        "lat": 37.0,
        "lon": -120.0,
        "features": { ndvi, ndwi, lst, ... },
        "primary_prediction": "Drought",
        "primary_confidence": 88.5,
        "auto_generate_tewi": true
      }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    engine = get_patent_engine()
    if not engine:
        return jsonify({"error": "Patent engine unavailable"}), 503

    try:
        lat        = float(data.get("lat", 0))
        lon        = float(data.get("lon", 0))
        features   = data.get("features", {})
        primary    = data.get("primary_prediction", "No Threat")
        confidence = float(data.get("primary_confidence", 70))

        # auto-generate time series if requested
        time_series = data.get("time_series")
        if not time_series and data.get("auto_generate_tewi") and primary != "No Threat":
            time_series = engine.tewi_engine.generate_synthetic_timeseries(
                features, primary, hours=72, readings=12
            )

        result = engine.full_analysis(
            lat=lat, lon=lon, features=features,
            time_series=time_series,
            primary_prediction=primary,
            primary_confidence=confidence,
            active_alerts=_alert_history[-10:]
        )
        result["claim_summary"] = {
            "claim_1_cascades_detected": len(result["claim_1_cascade"]),
            "claim_2_warnings_detected": len(result["claim_2_tewi"]),
            "claim_3_satellites_assigned": len(
                result["claim_3_dspr"]["assignments"]
                if result["claim_3_dspr"] else []
            ),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🛰️  ORION API (Patent Edition) starting...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
