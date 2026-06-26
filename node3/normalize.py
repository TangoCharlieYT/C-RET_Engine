"""
Node 3 — Data Handshake: Normalization & Input Assembly

This module standardizes all upstream data (FrameData from Node 1, TriageState
from Node 2) into fixed-size numerical vectors that the Context Autoencoder
and Safety Risk MLP can consume.

Input Sources:
    - FrameData.sensors:    speed, lat, lon, timestamp
    - FrameData.detections: variable-length list of {label, confidence, box}
    - TriageState:          anomaly_flags (low_conf, glare), kinematics (v_rel, crash_prob)

Output Vectors:
    - telemetry_vector:  (6,)  → Context Autoencoder
    - detection_vector:  (5,)  → Safety Risk MLP
    - label_vector:      (4,)  → Safety Risk MLP
    - anomaly_vector:    (3,)  → Safety Risk MLP
"""

import numpy as np
import math

# =============================================================================
# Constants — Normalization Ranges
# =============================================================================

# Speed: 0–150 km/h mapped to [0.0, 1.0]
SPEED_MIN = 0.0
SPEED_MAX = 150.0

# GPS: Delhi region defaults (configurable for other regions)
LAT_MIN = 28.0
LAT_MAX = 29.0
LON_MIN = 76.5
LON_MAX = 77.5

# Relative velocity: 0–50 m/s (~180 km/h) mapped to [0.0, 1.0]
V_REL_MAX = 50.0

# Time: seconds in a full day (for cyclical encoding)
SECONDS_PER_DAY = 86400.0

# Detection: max number of detections to keep (top-K by confidence)
MAX_DETECTIONS = 5

# Label vocabulary for one-hot encoding
LABEL_VOCAB = {
    "car": 0,
    "person": 1,
    "truck": 2,
}
NUM_LABEL_CLASSES = 4  # car, person, truck, other


# =============================================================================
# Telemetry Normalization (6-dim output)
# =============================================================================

def normalize_speed(speed_kmh: float) -> float:
    """Min-max normalize speed from [0, 150] km/h → [0.0, 1.0]."""
    return np.clip(speed_kmh / SPEED_MAX, 0.0, 1.0)


def normalize_latitude(lat: float) -> float:
    """Min-max normalize latitude from [LAT_MIN, LAT_MAX] → [0.0, 1.0]."""
    return np.clip((lat - LAT_MIN) / (LAT_MAX - LAT_MIN), 0.0, 1.0)


def normalize_longitude(lon: float) -> float:
    """Min-max normalize longitude from [LON_MIN, LON_MAX] → [0.0, 1.0]."""
    return np.clip((lon - LON_MIN) / (LON_MAX - LON_MIN), 0.0, 1.0)


def encode_time_cyclical(timestamp: float) -> tuple:
    """
    Convert a UNIX timestamp into cyclical sine/cosine encoding.

    This ensures that 23:59 and 00:01 are mathematically close
    (small angular distance on the unit circle), unlike raw seconds
    where they'd be 86,280 apart.

    Args:
        timestamp: UNIX timestamp (e.g., 1687372200.0)

    Returns:
        (time_sin, time_cos): both in range [-1.0, 1.0]
    """
    # Extract seconds since midnight from the UNIX timestamp
    seconds_since_midnight = timestamp % SECONDS_PER_DAY

    # Map to angle on the unit circle: 0 seconds → 0 radians, 86400 → 2π
    angle = (2.0 * math.pi * seconds_since_midnight) / SECONDS_PER_DAY

    return math.sin(angle), math.cos(angle)


def normalize_v_rel(v_rel: float) -> float:
    """Min-max normalize relative velocity from [0, 50] m/s → [0.0, 1.0]."""
    return np.clip(v_rel / V_REL_MAX, 0.0, 1.0)


def build_telemetry_vector(sensors: dict, timestamp: float, v_rel: float = 0.0) -> np.ndarray:
    """
    Assemble the full 6-dim normalized telemetry vector.

    Args:
        sensors:   dict with keys 'speed', 'lat', 'lon'
        timestamp: UNIX timestamp (float)
        v_rel:     relative velocity from Node 2 kinematics (m/s)

    Returns:
        np.ndarray of shape (6,), dtype float32
        [speed_norm, lat_norm, lon_norm, time_sin, time_cos, v_rel_norm]
    """
    speed_norm = normalize_speed(sensors.get("speed", 0.0))
    lat_norm = normalize_latitude(sensors.get("lat", LAT_MIN))
    lon_norm = normalize_longitude(sensors.get("lon", LON_MIN))
    time_sin, time_cos = encode_time_cyclical(timestamp)
    v_rel_norm = normalize_v_rel(v_rel)

    return np.array([
        speed_norm,
        lat_norm,
        lon_norm,
        time_sin,
        time_cos,
        v_rel_norm
    ], dtype=np.float32)


# =============================================================================
# YOLO Detection Standardization (5-dim confidences + 4-dim label one-hot)
# =============================================================================

def encode_label_onehot(label: str) -> np.ndarray:
    """
    One-hot encode a detection label.

    Known labels: 'car' → [1,0,0,0], 'person' → [0,1,0,0], 'truck' → [0,0,1,0]
    Anything else: 'other' → [0,0,0,1]

    Args:
        label: detection label string (case-insensitive)

    Returns:
        np.ndarray of shape (4,), dtype float32
    """
    onehot = np.zeros(NUM_LABEL_CLASSES, dtype=np.float32)
    idx = LABEL_VOCAB.get(label.lower(), NUM_LABEL_CLASSES - 1)
    onehot[idx] = 1.0
    return onehot


def pad_detections(detections: list) -> tuple:
    """
    Standardize a variable-length detection list into fixed-size vectors.

    Strategy:
        1. Sort detections by confidence (descending)
        2. Take the top MAX_DETECTIONS (5)
        3. If fewer than 5, zero-pad the remaining slots
        4. Extract confidence values → (5,) vector
        5. One-hot encode the top detection's label → (4,) vector

    Args:
        detections: list of dicts, each with keys 'label', 'confidence', 'box'
                    Example: [{"label": "car", "confidence": 0.88, "box": [x1, y1, x2, y2]}]

    Returns:
        (confidence_vector, label_vector):
            confidence_vector: np.ndarray of shape (5,), dtype float32
            label_vector:      np.ndarray of shape (4,), dtype float32 (one-hot of top detection)
    """
    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)

    # Extract top-K confidences, zero-pad if needed
    confidences = np.zeros(MAX_DETECTIONS, dtype=np.float32)
    for i in range(min(len(sorted_dets), MAX_DETECTIONS)):
        confidences[i] = np.clip(sorted_dets[i].get("confidence", 0.0), 0.0, 1.0)

    # One-hot encode the label of the highest-confidence detection
    if len(sorted_dets) > 0 and "label" in sorted_dets[0]:
        label_onehot = encode_label_onehot(sorted_dets[0]["label"])
    else:
        # No detections → zero vector (no dominant class)
        label_onehot = np.zeros(NUM_LABEL_CLASSES, dtype=np.float32)

    return confidences, label_onehot


# =============================================================================
# Anomaly Flag Encoding (3-dim output)
# =============================================================================

def build_anomaly_vector(triage_state: dict) -> np.ndarray:
    """
    Encode Node 2's anomaly flags and crash probability into a fixed-size vector.

    Args:
        triage_state: dict matching the TriageState contract:
            {
                "anomaly_flags": {"low_conf": bool, "glare": bool},
                "kinematics": {"v_rel": float, "crash_prob": float}
            }

    Returns:
        np.ndarray of shape (3,), dtype float32
        [low_conf_flag, glare_flag, crash_prob]
    """
    anomaly_flags = triage_state.get("anomaly_flags", {})
    kinematics = triage_state.get("kinematics", {})

    low_conf = 1.0 if anomaly_flags.get("low_conf", False) else 0.0
    glare = 1.0 if anomaly_flags.get("glare", False) else 0.0
    crash_prob = np.clip(kinematics.get("crash_prob", 0.0), 0.0, 1.0)

    return np.array([low_conf, glare, crash_prob], dtype=np.float32)


# =============================================================================
# Full Input Assembly
# =============================================================================

def assemble_autoencoder_input(sensors: dict, timestamp: float, v_rel: float = 0.0) -> np.ndarray:
    """
    Build the Context Autoencoder input vector.

    Returns:
        np.ndarray of shape (6,), dtype float32
    """
    return build_telemetry_vector(sensors, timestamp, v_rel)


def assemble_mlp_input(context_embedding: np.ndarray,
                       detections: list,
                       triage_state: dict) -> np.ndarray:
    """
    Build the Safety Risk MLP input vector by concatenating:
        - context_embedding (128-dim, from autoencoder encoder)
        - detection confidences (5-dim)
        - top label one-hot (4-dim)
        - anomaly flags + crash_prob (3-dim)

    Total: 128 + 5 + 4 + 3 = 140 dimensions

    Args:
        context_embedding: np.ndarray of shape (128,) from the trained encoder
        detections:        list of detection dicts from Node 1
        triage_state:      TriageState dict from Node 2

    Returns:
        np.ndarray of shape (140,), dtype float32
    """
    conf_vector, label_vector = pad_detections(detections)
    anomaly_vector = build_anomaly_vector(triage_state)

    return np.concatenate([
        context_embedding,   # 128
        conf_vector,         # 5
        label_vector,        # 4
        anomaly_vector       # 3
    ]).astype(np.float32)


def get_recommended_action(risk_score: float) -> str:
    """
    Map a continuous risk score [0.0, 1.0] to a discrete recommended action.

    Thresholds:
        0.00 – 0.25  →  "all_clear"
        0.25 – 0.50  →  "stay_alert"
        0.50 – 0.70  →  "reduce_speed"
        0.70 – 0.85  →  "slow_down"
        0.85 – 1.00  →  "brake_and_record"

    Args:
        risk_score: float in [0.0, 1.0]

    Returns:
        action string
    """
    if risk_score >= 0.85:
        return "brake_and_record"
    elif risk_score >= 0.70:
        return "slow_down"
    elif risk_score >= 0.50:
        return "reduce_speed"
    elif risk_score >= 0.25:
        return "stay_alert"
    else:
        return "all_clear"
