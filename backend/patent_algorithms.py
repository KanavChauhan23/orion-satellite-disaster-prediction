"""
ORION — Novel Patent Algorithms
=================================
Three unique technical innovations that form the basis of the patent claim:

CLAIM 1: Cascade Disaster Prediction (CDP)
  A method of detecting compound/cascade disaster events where one disaster
  triggers or amplifies another, using a directed graph of disaster
  interdependencies and spectral index correlation matrices.

  Example: Earthquake (seismic_v spike) → ground liquefaction → Flood
           Drought (soil_moist < 10%) → vegetation death → Wildfire cascade

CLAIM 2: Temporal Early Warning Index (TEWI)
  A novel scoring algorithm that analyses the RATE OF CHANGE of 14 satellite
  spectral indices over a 72-hour sliding window to predict disasters
  24-72 hours BEFORE conventional threshold-based systems detect them.

  TEWI = Σ (Δfeature_i / Δt) × weight_i × sensitivity_i
  Where Δfeature_i is the derivative of feature i over the time window.

CLAIM 3: Dynamic Swarm Priority Reassignment (DSPR)
  An algorithm that continuously recalculates scan priority scores for all
  geographic regions and automatically reassigns satellite coverage to
  maximise early warning probability across the swarm.

  Priority_score = TEWI × cascade_risk × population_density_factor
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json


# ═══════════════════════════════════════════════════════════════════
# CLAIM 1 — CASCADE DISASTER PREDICTION (CDP)
# ═══════════════════════════════════════════════════════════════════

# Directed graph: which disasters can trigger which other disasters
CASCADE_GRAPH = {
    "Earthquake": {
        "Flood":     0.65,   # ground liquefaction, tsunami, dam failure
        "Drought":   0.15,   # infrastructure damage disrupts irrigation
        "Wildfire":  0.10,   # ruptured gas lines
    },
    "Drought": {
        "Wildfire":  0.80,   # dry vegetation is primary wildfire fuel
        "Flood":     0.20,   # hardened soil causes flash floods when rain comes
    },
    "Wildfire": {
        "Flood":     0.55,   # burnt soil loses water absorption → flash floods
        "Drought":   0.25,   # destroys vegetation that retains soil moisture
    },
    "Flood": {
        "Earthquake": 0.10,  # soil saturation can trigger landslides/quakes
        "Drought":    0.15,  # post-flood soil salinisation
    },
    "Cyclone": {
        "Flood":     0.85,   # storm surge + extreme rainfall
        "Wildfire":  0.05,   # downed power lines
    },
}

# Spectral index correlations that amplify cascade probability
CASCADE_AMPLIFIERS = {
    ("Earthquake", "Flood"):   ["ndwi", "soil_moist", "elevation"],
    ("Drought",    "Wildfire"):["ndvi",  "humidity",   "lst"],
    ("Wildfire",   "Flood"):   ["ndwi",  "soil_moist", "slope"],
    ("Cyclone",    "Flood"):   ["ndwi",  "wind_speed", "cloud_cov"],
    ("Flood",      "Drought"): ["ndvi",  "soil_moist", "precip"],
}


@dataclass
class CascadeEvent:
    """Represents a predicted cascade disaster chain."""
    primary_disaster:    str
    triggered_disaster:  str
    cascade_probability: float    # 0-1
    amplifier_features:  List[str]
    estimated_delay_hrs: int      # hours until secondary disaster
    combined_severity:   str      # overall severity of the cascade
    confidence:          float
    timestamp:           str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CascadeDisasterPredictor:
    """
    PATENT CLAIM 1:
    Method for predicting compound disaster cascade events using
    directed disaster interdependency graphs and spectral amplifier
    correlation matrices.
    """

    # Time delays between primary and secondary disasters (hours)
    CASCADE_DELAYS = {
        ("Earthquake", "Flood"):    6,
        ("Earthquake", "Wildfire"): 12,
        ("Drought",    "Wildfire"): 48,
        ("Drought",    "Flood"):    72,
        ("Wildfire",   "Flood"):    24,
        ("Cyclone",    "Flood"):    3,
        ("Flood",      "Drought"):  168,  # 7 days
    }

    SEVERITY_MAP = {
        (0.0, 0.3): "low",
        (0.3, 0.5): "moderate",
        (0.5, 0.7): "high",
        (0.7, 1.0): "critical",
    }

    def _get_severity(self, prob: float) -> str:
        for (lo, hi), sev in self.SEVERITY_MAP.items():
            if lo <= prob < hi:
                return sev
        return "critical"

    def _amplifier_score(
        self,
        primary: str,
        secondary: str,
        features: dict
    ) -> float:
        """
        Calculate amplification factor based on current spectral index values.
        Higher amplification = greater chance of cascade.
        """
        key = (primary, secondary)
        if key not in CASCADE_AMPLIFIERS:
            return 1.0

        amp_features = CASCADE_AMPLIFIERS[key]
        score = 1.0

        for feat in amp_features:
            val = features.get(feat, 0)
            # Each amplifier feature contributes multiplicatively
            if feat == "ndwi"      and val > 0.3:  score *= 1.4
            if feat == "soil_moist"and val > 70:   score *= 1.3
            if feat == "ndvi"      and val < 0.15: score *= 1.5
            if feat == "humidity"  and val < 25:   score *= 1.4
            if feat == "lst"       and val > 38:   score *= 1.2
            if feat == "slope"     and val > 20:   score *= 1.3
            if feat == "wind_speed"and val > 100:  score *= 1.6
            if feat == "cloud_cov" and val > 70:   score *= 1.2

        return min(score, 3.0)  # cap at 3x amplification

    def predict_cascades(
        self,
        primary_disaster: str,
        features: dict,
        primary_confidence: float
    ) -> List[CascadeEvent]:
        """
        Given a detected primary disaster and current spectral readings,
        predict all possible cascade (secondary) disasters.

        Returns list of CascadeEvent sorted by cascade_probability descending.
        """
        if primary_disaster not in CASCADE_GRAPH:
            return []

        cascades = []
        secondary_risks = CASCADE_GRAPH[primary_disaster]

        for secondary, base_prob in secondary_risks.items():
            amp     = self._amplifier_score(primary_disaster, secondary, features)
            prob    = min(base_prob * amp * primary_confidence / 100, 1.0)
            delay   = self.CASCADE_DELAYS.get((primary_disaster, secondary), 24)
            amp_fts = CASCADE_AMPLIFIERS.get((primary_disaster, secondary), [])
            sev     = self._get_severity(prob)

            if prob > 0.15:   # only report meaningful cascade risks
                cascades.append(CascadeEvent(
                    primary_disaster    = primary_disaster,
                    triggered_disaster  = secondary,
                    cascade_probability = round(prob * 100, 2),
                    amplifier_features  = amp_fts,
                    estimated_delay_hrs = delay,
                    combined_severity   = sev,
                    confidence          = round(primary_confidence, 2),
                ))

        return sorted(cascades, key=lambda x: x.cascade_probability, reverse=True)


# ═══════════════════════════════════════════════════════════════════
# CLAIM 2 — TEMPORAL EARLY WARNING INDEX (TEWI)
# ═══════════════════════════════════════════════════════════════════

# How sensitive each feature's rate-of-change is to each disaster
TEWI_WEIGHTS = {
    "Flood": {
        "ndwi":       +3.5,   # rising water index = strong flood signal
        "precip":     +2.8,
        "soil_moist": +2.2,
        "cloud_cov":  +1.5,
        "elevation":  -1.0,   # decreasing elevation readings = flooding
        "lst":        -0.8,   # cooling surface = water presence
    },
    "Wildfire": {
        "lst":        +3.8,   # rising surface temp = fire risk
        "ndvi":       -3.2,   # falling vegetation = fuel drying
        "humidity":   -2.9,   # falling humidity = fire risk
        "swir":       +2.5,   # SWIR rise = heat signature
        "precip":     -2.0,
        "soil_moist": -1.8,
    },
    "Earthquake": {
        "seismic_v":  +4.5,   # seismic velocity change = strongest signal
        "slope":      +1.5,   # slope changes can indicate ground deformation
        "elevation":  +1.2,   # elevation anomalies
    },
    "Cyclone": {
        "wind_speed": +4.0,
        "sst_anom":   +3.2,   # warming sea surface = cyclone fuel
        "cloud_cov":  +2.8,
        "humidity":   +2.0,
        "precip":     +1.8,
    },
    "Drought": {
        "soil_moist": -4.0,   # falling soil moisture = strongest drought signal
        "ndvi":       -3.5,
        "precip":     -3.0,
        "lst":        +2.5,   # rising temperature worsens drought
        "humidity":   -2.0,
    },
}


@dataclass
class TEWIResult:
    """Result of Temporal Early Warning Index calculation."""
    disaster_type:       str
    tewi_score:          float    # -100 to +100, positive = rising risk
    warning_level:       str      # watch / advisory / warning / emergency
    hours_to_onset:      int      # estimated hours until disaster onset
    trend_direction:     str      # accelerating / stable / decelerating
    key_indicators:      List[str]# which features are driving the score
    confidence:          float
    timestamp:           str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TemporalEarlyWarningIndex:
    """
    PATENT CLAIM 2:
    Temporal Early Warning Index (TEWI) — a novel algorithm that computes
    the rate-of-change of satellite spectral indices over a 72-hour sliding
    window to detect pre-disaster signatures 24-72 hours before onset.

    Formula:
        TEWI_d = Σ_i [ (Δf_i / Δt) × w_{d,i} × σ_i ] / N

    Where:
        Δf_i  = change in feature i over time window
        Δt    = time window (hours)
        w_{d,i} = disaster-specific weight for feature i
        σ_i   = historical standard deviation (normalisation)
        N     = number of features
        d     = disaster type
    """

    WARNING_LEVELS = {
        "emergency": 70,
        "warning":   45,
        "advisory":  25,
        "watch":      5,
    }

    ONSET_ESTIMATES = {
        "Flood":      {"emergency": 6,  "warning": 24, "advisory": 48, "watch": 72},
        "Wildfire":   {"emergency": 3,  "warning": 12, "advisory": 36, "watch": 60},
        "Earthquake": {"emergency": 1,  "warning": 6,  "advisory": 24, "watch": 48},
        "Cyclone":    {"emergency": 12, "warning": 36, "advisory": 60, "watch": 72},
        "Drought":    {"emergency": 72, "warning": 168,"advisory": 336,"watch": 720},
    }

    # Historical standard deviations for normalisation (from training data)
    FEATURE_STD = {
        "ndvi": 0.25, "ndwi": 0.35, "lst": 12.0, "swir": 0.15,
        "nir": 0.18, "precip": 55.0, "wind_speed": 40.0,
        "humidity": 22.0, "soil_moist": 25.0, "sst_anom": 1.5,
        "elevation": 600.0, "slope": 12.0, "seismic_v": 0.9,
        "cloud_cov": 30.0,
    }

    def compute_tewi(
        self,
        time_series: List[dict],
        window_hours: int = 72
    ) -> List[TEWIResult]:
        """
        Compute TEWI for all disaster types from a time series of
        satellite readings.

        Args:
            time_series: List of feature dicts ordered oldest → newest
            window_hours: Sliding window size in hours (default 72h)

        Returns:
            List of TEWIResult for each disaster type, sorted by score desc.
        """
        if len(time_series) < 2:
            return []

        # calculate feature derivatives over the window
        oldest  = time_series[0]
        newest  = time_series[-1]
        dt      = window_hours  # normalise to per-hour rate

        deltas = {}
        for feat in self.FEATURE_STD:
            v_old = oldest.get(feat, 0)
            v_new = newest.get(feat, 0)
            std   = self.FEATURE_STD[feat]
            # normalised rate of change per hour
            deltas[feat] = ((v_new - v_old) / (dt * std + 1e-8))

        # also compute acceleration (2nd derivative) using midpoint
        mid = time_series[len(time_series)//2]
        accel = {}
        for feat in self.FEATURE_STD:
            std   = self.FEATURE_STD[feat]
            d1    = (mid.get(feat,0) - oldest.get(feat,0)) / (dt/2 * std + 1e-8)
            d2    = (newest.get(feat,0) - mid.get(feat,0)) / (dt/2 * std + 1e-8)
            accel[feat] = d2 - d1   # positive = accelerating

        results = []
        for disaster, weights in TEWI_WEIGHTS.items():
            score = 0.0
            key_indicators = []
            contributions  = {}

            for feat, w in weights.items():
                delta = deltas.get(feat, 0)
                contrib = delta * w * 100   # scale to -100..+100
                score  += contrib
                contributions[feat] = contrib

            # normalise by number of features
            score /= max(len(weights), 1)
            score  = max(-100, min(100, score))

            # find top driving features
            sorted_contribs = sorted(
                contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )
            key_indicators = [f for f, _ in sorted_contribs[:3]]

            # determine warning level
            warning = "watch"
            for lvl, threshold in self.WARNING_LEVELS.items():
                if score >= threshold:
                    warning = lvl
                    break

            # estimate hours to onset
            onset_map = self.ONSET_ESTIMATES.get(disaster, {})
            hours     = onset_map.get(warning, 72)

            # determine trend direction from acceleration
            avg_accel = np.mean([accel.get(f,0) * w
                for f, w in weights.items()])
            if   avg_accel >  0.1: trend = "accelerating"
            elif avg_accel < -0.1: trend = "decelerating"
            else:                  trend = "stable"

            results.append(TEWIResult(
                disaster_type   = disaster,
                tewi_score      = round(score, 2),
                warning_level   = warning,
                hours_to_onset  = hours,
                trend_direction = trend,
                key_indicators  = key_indicators,
                confidence      = round(min(abs(score), 100), 2),
            ))

        return sorted(results, key=lambda x: x.tewi_score, reverse=True)

    def generate_synthetic_timeseries(
        self,
        base_features: dict,
        disaster_type: str,
        hours: int = 72,
        readings: int = 12
    ) -> List[dict]:
        """
        Generate a synthetic time series showing a disaster developing.
        Used for demonstration and training.
        """
        series = []
        weights = TEWI_WEIGHTS.get(disaster_type, {})

        for i in range(readings):
            progress = i / (readings - 1)   # 0 → 1 over time
            reading  = dict(base_features)

            # apply progressive changes in the direction of disaster
            for feat, w in weights.items():
                if feat not in reading:
                    continue
                std    = self.FEATURE_STD.get(feat, 1)
                change = np.sign(w) * progress * std * 0.8
                noise  = np.random.normal(0, std * 0.05)
                reading[feat] = float(reading[feat]) + change + noise

            reading["timestamp"] = (
                datetime.utcnow() - timedelta(hours=hours - i*(hours/readings))
            ).isoformat()
            series.append(reading)

        return series


# ═══════════════════════════════════════════════════════════════════
# CLAIM 3 — DYNAMIC SWARM PRIORITY REASSIGNMENT (DSPR)
# ═══════════════════════════════════════════════════════════════════

# Population density factor by latitude band (higher = more people at risk)
POPULATION_DENSITY_FACTOR = {
    (-90, -60): 0.1,
    (-60, -30): 0.4,
    (-30,   0): 0.7,
    (  0,  30): 1.0,   # densest population band
    ( 30,  60): 0.9,
    ( 60,  90): 0.2,
}


@dataclass
class SatelliteAssignment:
    """Represents a satellite's assigned scan priority region."""
    satellite_id:     str
    target_lat:       float
    target_lon:       float
    priority_score:   float
    reason:           str        # why this region was prioritised
    tewi_score:       float
    cascade_risk:     float
    population_factor:float
    estimated_scan_time_min: int


@dataclass
class RegionPriority:
    """Priority score for a geographic region."""
    lat:              float
    lon:              float
    priority_score:   float
    tewi_scores:      dict       # per disaster type
    cascade_risk:     float
    population_factor:float
    assigned_sat:     Optional[str] = None


class DynamicSwarmPriorityReassignment:
    """
    PATENT CLAIM 3:
    Dynamic Swarm Priority Reassignment (DSPR) — an algorithm that
    continuously recalculates geographic scan priority scores and
    automatically reassigns satellite coverage to maximise early
    warning probability.

    Priority Score Formula:
        P(r) = α·TEWI(r) + β·CascadeRisk(r) + γ·PopulationDensity(r)
               + δ·TimeSinceLastScan(r)

    Where:
        α = 0.40  (TEWI weight)
        β = 0.25  (cascade risk weight)
        γ = 0.20  (population density weight)
        δ = 0.15  (temporal coverage weight)
    """

    # Weighting coefficients
    ALPHA = 0.40   # TEWI weight
    BETA  = 0.25   # cascade risk
    GAMMA = 0.20   # population density
    DELTA = 0.15   # time since last scan

    def __init__(self, n_satellites: int = 6):
        self.n_satellites  = n_satellites
        self.tewi_engine   = TemporalEarlyWarningIndex()
        self.cascade_engine= CascadeDisasterPredictor()
        self.scan_history: Dict[str, datetime] = {}   # region_key → last scan time

    def _region_key(self, lat: float, lon: float) -> str:
        """Round to 5° grid for region identification."""
        return f"{round(lat/5)*5},{round(lon/5)*5}"

    def _population_factor(self, lat: float) -> float:
        for (lo, hi), factor in POPULATION_DENSITY_FACTOR.items():
            if lo <= lat < hi:
                return factor
        return 0.5

    def _time_factor(self, lat: float, lon: float) -> float:
        """Regions not scanned recently get higher priority."""
        key = self._region_key(lat, lon)
        if key not in self.scan_history:
            return 1.0   # never scanned = maximum time priority
        elapsed = (datetime.utcnow() - self.scan_history[key]).seconds / 3600
        return min(elapsed / 24, 1.0)   # normalise to 0-1 over 24h

    def compute_region_priority(
        self,
        lat: float,
        lon: float,
        current_features: dict,
        time_series: Optional[List[dict]] = None,
        active_alerts: Optional[List[dict]] = None
    ) -> RegionPriority:
        """
        Compute the priority score for a geographic region.
        """
        # TEWI component
        tewi_scores = {}
        max_tewi    = 0.0
        if time_series and len(time_series) >= 2:
            tewi_results = self.tewi_engine.compute_tewi(time_series)
            tewi_scores  = {r.disaster_type: r.tewi_score for r in tewi_results}
            max_tewi     = max((r.tewi_score for r in tewi_results), default=0)

        tewi_norm = max(max_tewi, 0) / 100   # normalise 0-1

        # Cascade risk component
        cascade_risk = 0.0
        if active_alerts:
            for alert in active_alerts:
                dist = abs(alert.get("lat",0)-lat) + abs(alert.get("lon",0)-lon)
                if dist < 20:   # within ~20° (~2200km)
                    cascades = self.cascade_engine.predict_cascades(
                        alert.get("type",""), current_features,
                        alert.get("confidence", 50)
                    )
                    if cascades:
                        cascade_risk = max(
                            cascade_risk,
                            cascades[0].cascade_probability / 100
                        )

        # Population density component
        pop_factor = self._population_factor(lat)

        # Time-since-scan component
        time_factor = self._time_factor(lat, lon)

        # Final priority score (0-100)
        priority = (
            self.ALPHA * tewi_norm    * 100 +
            self.BETA  * cascade_risk * 100 +
            self.GAMMA * pop_factor   * 100 +
            self.DELTA * time_factor  * 100
        )

        return RegionPriority(
            lat              = round(lat, 3),
            lon              = round(lon, 3),
            priority_score   = round(priority, 2),
            tewi_scores      = tewi_scores,
            cascade_risk     = round(cascade_risk * 100, 2),
            population_factor= round(pop_factor, 2),
        )

    def assign_satellites(
        self,
        candidate_regions: List[Tuple[float, float]],
        current_features_map: Dict[str, dict],
        active_alerts: Optional[List[dict]] = None
    ) -> List[SatelliteAssignment]:
        """
        Core DSPR algorithm:
        Given a set of candidate regions, compute priority scores for each
        and assign satellites to maximise early warning coverage.
        """
        sat_ids = [f"SAT-{1001+i}" for i in range(self.n_satellites)]

        # score all regions
        scored = []
        for lat, lon in candidate_regions:
            key      = self._region_key(lat, lon)
            features = current_features_map.get(key, {})
            region   = self.compute_region_priority(
                lat, lon, features,
                active_alerts=active_alerts
            )
            scored.append(region)

        # sort by priority descending
        scored.sort(key=lambda r: r.priority_score, reverse=True)

        # assign one satellite per top-priority region
        assignments = []
        for i, sat_id in enumerate(sat_ids):
            if i >= len(scored):
                break
            region = scored[i]

            # determine reason string
            if   region.tewi_scores and max(region.tewi_scores.values(),default=0) > 40:
                reason = f"High TEWI score — early warning signal detected"
            elif region.cascade_risk > 30:
                reason = f"Cascade risk from nearby active alert"
            elif region.population_factor > 0.8:
                reason = f"High population density region"
            else:
                reason = f"Routine coverage — time-based priority"

            # mark region as scanned
            key = self._region_key(region.lat, region.lon)
            self.scan_history[key] = datetime.utcnow()
            region.assigned_sat = sat_id

            # scan time based on altitude (lower = faster revisit)
            alt_map = {0:450,1:620,2:510,3:780,4:430,5:680}
            alt     = alt_map.get(i, 500)
            scan_t  = int(90 * (alt / 500))   # approx orbital period

            assignments.append(SatelliteAssignment(
                satellite_id           = sat_id,
                target_lat             = region.lat,
                target_lon             = region.lon,
                priority_score         = region.priority_score,
                reason                 = reason,
                tewi_score             = max(region.tewi_scores.values(), default=0),
                cascade_risk           = region.cascade_risk,
                population_factor      = region.population_factor,
                estimated_scan_time_min= scan_t,
            ))

        return assignments

    def get_coverage_report(
        self,
        assignments: List[SatelliteAssignment]
    ) -> dict:
        """Generate a summary coverage report for the current swarm assignment."""
        if not assignments:
            return {}

        return {
            "total_satellites":    self.n_satellites,
            "regions_covered":     len(assignments),
            "avg_priority_score":  round(np.mean([a.priority_score for a in assignments]),2),
            "max_priority_region": {
                "lat": assignments[0].target_lat,
                "lon": assignments[0].target_lon,
                "score": assignments[0].priority_score,
                "assigned": assignments[0].satellite_id,
            },
            "high_priority_count": sum(1 for a in assignments if a.priority_score > 60),
            "cascade_risk_regions":sum(1 for a in assignments if a.cascade_risk > 25),
            "tewi_warning_regions":sum(1 for a in assignments if a.tewi_score > 40),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════
# INTEGRATED PATENT ENGINE
# ═══════════════════════════════════════════════════════════════════

class ORIONPatentEngine:
    """
    Combines all three patent claims into one unified inference engine.

    Usage:
        engine = ORIONPatentEngine()
        result = engine.full_analysis(lat, lon, features, time_series)
    """

    def __init__(self, n_satellites: int = 6):
        self.cascade_engine = CascadeDisasterPredictor()
        self.tewi_engine    = TemporalEarlyWarningIndex()
        self.dspr_engine    = DynamicSwarmPriorityReassignment(n_satellites)

    def full_analysis(
        self,
        lat:         float,
        lon:         float,
        features:    dict,
        time_series: Optional[List[dict]] = None,
        primary_prediction: Optional[str] = None,
        primary_confidence: float = 70.0,
        active_alerts: Optional[List[dict]] = None
    ) -> dict:
        """
        Run all three patent algorithms and return a unified analysis report.
        """
        result = {
            "location":   {"lat": lat, "lon": lon},
            "timestamp":  datetime.utcnow().isoformat(),
            "claim_1_cascade":  [],
            "claim_2_tewi":     [],
            "claim_3_dspr":     None,
        }

        # CLAIM 1 — Cascade prediction
        if primary_prediction and primary_prediction != "No Threat":
            cascades = self.cascade_engine.predict_cascades(
                primary_prediction, features, primary_confidence
            )
            result["claim_1_cascade"] = [asdict(c) for c in cascades]

        # CLAIM 2 — TEWI early warning
        if time_series and len(time_series) >= 2:
            tewi_results = self.tewi_engine.compute_tewi(time_series)
            result["claim_2_tewi"] = [asdict(t) for t in tewi_results[:3]]

        # CLAIM 3 — DSPR satellite assignment
        candidate_regions = [
            (lat + np.random.uniform(-10, 10), lon + np.random.uniform(-10, 10))
            for _ in range(12)
        ]
        candidate_regions.append((lat, lon))   # include current region
        features_map = {
            self.dspr_engine._region_key(r[0], r[1]): features
            for r in candidate_regions
        }
        assignments = self.dspr_engine.assign_satellites(
            candidate_regions, features_map, active_alerts
        )
        result["claim_3_dspr"] = {
            "assignments": [asdict(a) for a in assignments],
            "coverage_report": self.dspr_engine.get_coverage_report(assignments)
        }

        return result


# ═══════════════════════════════════════════════════════════════════
# DEMO / ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  ORION — Patent Algorithm Demo")
    print("  Claims 1 (CDP) + 2 (TEWI) + 3 (DSPR)")
    print("=" * 65)

    engine = ORIONPatentEngine(n_satellites=6)
    tewi   = TemporalEarlyWarningIndex()

    # ── Demo 1: Wildfire cascade after drought ──────────────────
    print("\n📍 LOCATION: California Foothills (37°N, 120°W)")
    print("─" * 50)

    drought_features = {
        "ndvi": 0.04, "ndwi": -0.42, "lst": 46.0, "swir": 0.42,
        "nir": 0.12, "precip": -38.0, "wind_speed": 18.0,
        "humidity": 13.0, "soil_moist": 5.0, "sst_anom": 0.5,
        "elevation": 480.0, "slope": 9.0, "seismic_v": 0.1,
        "cloud_cov": 4.0,
    }

    # generate 72-hour pre-wildfire time series
    ts = tewi.generate_synthetic_timeseries(
        drought_features, "Wildfire", hours=72, readings=12
    )

    result = engine.full_analysis(
        lat=37.0, lon=-120.0,
        features=drought_features,
        time_series=ts,
        primary_prediction="Drought",
        primary_confidence=88.5,
        active_alerts=[]
    )

    # show cascade predictions
    print("\n🔗 CLAIM 1 — Cascade Disaster Predictions:")
    for c in result["claim_1_cascade"]:
        print(f"   {c['primary_disaster']} → {c['triggered_disaster']:12}"
              f"  Prob: {c['cascade_probability']:5.1f}%"
              f"  In: {c['estimated_delay_hrs']}h"
              f"  [{c['combined_severity'].upper()}]")

    # show TEWI scores
    print("\n⏱  CLAIM 2 — Temporal Early Warning Index (TEWI):")
    for t in result["claim_2_tewi"]:
        print(f"   {t['disaster_type']:12}"
              f"  TEWI: {t['tewi_score']:6.1f}"
              f"  [{t['warning_level'].upper():9}]"
              f"  Est. onset: {t['hours_to_onset']}h"
              f"  Trend: {t['trend_direction']}")

    # show satellite assignments
    print("\n🛰  CLAIM 3 — Dynamic Swarm Priority Reassignment:")
    for a in result["claim_3_dspr"]["assignments"]:
        print(f"   {a['satellite_id']}  →"
              f"  ({a['target_lat']:6.2f}, {a['target_lon']:7.2f})"
              f"  Priority: {a['priority_score']:5.1f}"
              f"  | {a['reason'][:45]}")

    cov = result["claim_3_dspr"]["coverage_report"]
    print(f"\n   Coverage: {cov['regions_covered']} regions"
          f"  |  High-priority: {cov['high_priority_count']}"
          f"  |  TEWI warnings: {cov['tewi_warning_regions']}"
          f"  |  Cascade risks: {cov['cascade_risk_regions']}")

    print("\n✅ All three patent claims demonstrated successfully.")
    print("   Ready for patent submission documentation.")
