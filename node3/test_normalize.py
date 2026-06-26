"""
Unit Tests for Node 3 — Data Handshake (Phase 1)

Tests cover:
    - Telemetry normalization (speed, lat, lon, cyclical time, v_rel)
    - Detection padding (top-5, zero-pad, empty list)
    - Label one-hot encoding (known labels, unknown labels)
    - Anomaly flag encoding (all combos)
    - Full input assembly (shape validation)
    - Action mapping thresholds
    - Edge cases (midnight wraparound, zero speed, out-of-range values)
"""

import sys
import os
import math
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.normalize import (
    normalize_speed,
    normalize_latitude,
    normalize_longitude,
    encode_time_cyclical,
    normalize_v_rel,
    build_telemetry_vector,
    encode_label_onehot,
    pad_detections,
    build_anomaly_vector,
    assemble_autoencoder_input,
    assemble_mlp_input,
    get_recommended_action,
    SPEED_MAX, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, V_REL_MAX,
    MAX_DETECTIONS, NUM_LABEL_CLASSES,
)

# Track test results
_passed = 0
_failed = 0


def assert_close(actual, expected, tol=1e-5, msg=""):
    global _passed, _failed
    if abs(actual - expected) <= tol:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {msg} — expected {expected}, got {actual}")


def assert_equal(actual, expected, msg=""):
    global _passed, _failed
    if actual == expected:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {msg} — expected {expected}, got {actual}")


def assert_shape(arr, expected_shape, msg=""):
    global _passed, _failed
    if arr.shape == expected_shape:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {msg} — expected shape {expected_shape}, got {arr.shape}")


def assert_dtype(arr, expected_dtype, msg=""):
    global _passed, _failed
    if arr.dtype == expected_dtype:
        _passed += 1
    else:
        _failed += 1
        print(f"  FAIL: {msg} — expected dtype {expected_dtype}, got {arr.dtype}")


# =========================================================================
# Test: Speed Normalization
# =========================================================================
def test_speed_normalization():
    print("\n[TEST] Speed Normalization")

    # Normal values
    assert_close(normalize_speed(0.0), 0.0, msg="speed=0")
    assert_close(normalize_speed(75.0), 0.5, msg="speed=75")
    assert_close(normalize_speed(150.0), 1.0, msg="speed=150")

    # Clamping: above max
    assert_close(normalize_speed(200.0), 1.0, msg="speed=200 (clamp)")

    # Clamping: negative (shouldn't happen, but defensive)
    assert_close(normalize_speed(-10.0), 0.0, msg="speed=-10 (clamp)")


# =========================================================================
# Test: Latitude / Longitude Normalization
# =========================================================================
def test_gps_normalization():
    print("\n[TEST] GPS Normalization")

    assert_close(normalize_latitude(28.0), 0.0, msg="lat=28.0 (min)")
    assert_close(normalize_latitude(28.5), 0.5, msg="lat=28.5 (mid)")
    assert_close(normalize_latitude(29.0), 1.0, msg="lat=29.0 (max)")
    assert_close(normalize_latitude(27.0), 0.0, msg="lat=27.0 (clamp low)")
    assert_close(normalize_latitude(30.0), 1.0, msg="lat=30.0 (clamp high)")

    assert_close(normalize_longitude(76.5), 0.0, msg="lon=76.5 (min)")
    assert_close(normalize_longitude(77.0), 0.5, msg="lon=77.0 (mid)")
    assert_close(normalize_longitude(77.5), 1.0, msg="lon=77.5 (max)")


# =========================================================================
# Test: Cyclical Time Encoding
# =========================================================================
def test_cyclical_time():
    print("\n[TEST] Cyclical Time Encoding")

    # Midnight (0 seconds into day) → sin=0, cos=1
    ts_midnight = 0.0  # UNIX epoch is midnight UTC
    sin_val, cos_val = encode_time_cyclical(ts_midnight)
    assert_close(sin_val, 0.0, tol=1e-4, msg="midnight sin")
    assert_close(cos_val, 1.0, tol=1e-4, msg="midnight cos")

    # 6 AM (21600 seconds) → sin=1, cos=0  (quarter circle)
    ts_6am = 21600.0
    sin_val, cos_val = encode_time_cyclical(ts_6am)
    assert_close(sin_val, 1.0, tol=1e-4, msg="6AM sin")
    assert_close(cos_val, 0.0, tol=1e-4, msg="6AM cos")

    # Noon (43200 seconds) → sin=0, cos=-1 (half circle)
    ts_noon = 43200.0
    sin_val, cos_val = encode_time_cyclical(ts_noon)
    assert_close(sin_val, 0.0, tol=1e-4, msg="noon sin")
    assert_close(cos_val, -1.0, tol=1e-4, msg="noon cos")

    # 11:59 PM and 00:01 AM should be close (wraparound test)
    ts_2359 = 86340.0   # 23:59
    ts_0001 = 60.0      # 00:01
    sin_a, cos_a = encode_time_cyclical(ts_2359)
    sin_b, cos_b = encode_time_cyclical(ts_0001)
    # Angular distance should be small (2 minutes = 2/1440 of circle ≈ 0.0087 radians)
    angular_dist = math.sqrt((sin_a - sin_b)**2 + (cos_a - cos_b)**2)
    assert_close(angular_dist, 0.0, tol=0.02, msg="23:59 vs 00:01 proximity")


# =========================================================================
# Test: Relative Velocity Normalization
# =========================================================================
def test_v_rel_normalization():
    print("\n[TEST] Relative Velocity Normalization")

    assert_close(normalize_v_rel(0.0), 0.0, msg="v_rel=0")
    assert_close(normalize_v_rel(25.0), 0.5, msg="v_rel=25")
    assert_close(normalize_v_rel(50.0), 1.0, msg="v_rel=50")
    assert_close(normalize_v_rel(80.0), 1.0, msg="v_rel=80 (clamp)")


# =========================================================================
# Test: Telemetry Vector Assembly
# =========================================================================
def test_telemetry_vector():
    print("\n[TEST] Telemetry Vector Assembly")

    sensors = {"speed": 90.0, "lat": 28.5, "lon": 77.0}
    timestamp = 43200.0  # noon
    v_rel = 10.0

    vec = build_telemetry_vector(sensors, timestamp, v_rel)

    assert_shape(vec, (6,), msg="telemetry shape")
    assert_dtype(vec, np.float32, msg="telemetry dtype")
    assert_close(vec[0], 90.0 / 150.0, msg="speed component")
    assert_close(vec[1], 0.5, msg="lat component")
    assert_close(vec[2], 0.5, msg="lon component")
    assert_close(vec[5], 10.0 / 50.0, msg="v_rel component")

    # Missing keys should default safely
    vec_empty = build_telemetry_vector({}, 0.0)
    assert_shape(vec_empty, (6,), msg="empty sensors shape")
    assert_close(vec_empty[0], 0.0, msg="default speed")


# =========================================================================
# Test: Label One-Hot Encoding
# =========================================================================
def test_label_encoding():
    print("\n[TEST] Label One-Hot Encoding")

    car = encode_label_onehot("car")
    assert_shape(car, (4,), msg="car shape")
    assert_close(car[0], 1.0, msg="car index 0")
    assert_close(car[1], 0.0, msg="car index 1")

    person = encode_label_onehot("person")
    assert_close(person[1], 1.0, msg="person index 1")

    truck = encode_label_onehot("truck")
    assert_close(truck[2], 1.0, msg="truck index 2")

    # Unknown label → "other" slot (index 3)
    bicycle = encode_label_onehot("bicycle")
    assert_close(bicycle[3], 1.0, msg="unknown→other index 3")

    # Case insensitivity
    car_upper = encode_label_onehot("CAR")
    assert_close(car_upper[0], 1.0, msg="CAR (uppercase) → car")


# =========================================================================
# Test: Detection Padding
# =========================================================================
def test_detection_padding():
    print("\n[TEST] Detection Padding")

    # Normal case: 3 detections → top-3 filled, last 2 zero
    dets = [
        {"label": "car", "confidence": 0.88, "box": [10, 20, 100, 200]},
        {"label": "person", "confidence": 0.72, "box": [50, 60, 150, 250]},
        {"label": "truck", "confidence": 0.65, "box": [200, 100, 400, 300]},
    ]
    conf, label = pad_detections(dets)
    assert_shape(conf, (5,), msg="conf shape (3 dets)")
    assert_shape(label, (4,), msg="label shape (3 dets)")
    assert_close(conf[0], 0.88, msg="top conf")
    assert_close(conf[1], 0.72, msg="2nd conf")
    assert_close(conf[2], 0.65, msg="3rd conf")
    assert_close(conf[3], 0.0, msg="4th conf (pad)")
    assert_close(conf[4], 0.0, msg="5th conf (pad)")
    assert_close(label[0], 1.0, msg="top label is car")

    # Empty detections → all zeros
    conf_e, label_e = pad_detections([])
    assert_shape(conf_e, (5,), msg="conf shape (empty)")
    assert_close(conf_e[0], 0.0, msg="empty conf[0]")
    assert_close(label_e.sum(), 0.0, msg="empty label sum")

    # Overflow case: 8 detections → only top-5 kept
    dets_many = [{"label": "car", "confidence": i * 0.1, "box": [0, 0, 0, 0]} for i in range(8)]
    conf_m, _ = pad_detections(dets_many)
    assert_close(conf_m[0], 0.7, msg="overflow top conf")
    assert_close(conf_m[4], 0.3, msg="overflow 5th conf")


# =========================================================================
# Test: Anomaly Vector
# =========================================================================
def test_anomaly_vector():
    print("\n[TEST] Anomaly Vector")

    triage = {
        "anomaly_flags": {"low_conf": True, "glare": False},
        "kinematics": {"v_rel": 15.0, "crash_prob": 0.42}
    }
    vec = build_anomaly_vector(triage)
    assert_shape(vec, (3,), msg="anomaly shape")
    assert_dtype(vec, np.float32, msg="anomaly dtype")
    assert_close(vec[0], 1.0, msg="low_conf=True")
    assert_close(vec[1], 0.0, msg="glare=False")
    assert_close(vec[2], 0.42, msg="crash_prob")

    # Missing keys → defaults
    vec_empty = build_anomaly_vector({})
    assert_close(vec_empty[0], 0.0, msg="default low_conf")
    assert_close(vec_empty[1], 0.0, msg="default glare")
    assert_close(vec_empty[2], 0.0, msg="default crash_prob")


# =========================================================================
# Test: Full MLP Input Assembly
# =========================================================================
def test_mlp_assembly():
    print("\n[TEST] Full MLP Input Assembly")

    # Simulate a 128-dim context embedding
    fake_embedding = np.random.randn(128).astype(np.float32)

    detections = [
        {"label": "car", "confidence": 0.9, "box": [0, 0, 0, 0]},
        {"label": "person", "confidence": 0.5, "box": [0, 0, 0, 0]},
    ]

    triage = {
        "anomaly_flags": {"low_conf": False, "glare": True},
        "kinematics": {"v_rel": 20.0, "crash_prob": 0.88}
    }

    mlp_input = assemble_mlp_input(fake_embedding, detections, triage)

    # Must be exactly 140 dimensions
    assert_shape(mlp_input, (140,), msg="MLP input shape")
    assert_dtype(mlp_input, np.float32, msg="MLP input dtype")

    # First 128 should match the embedding
    assert_close(float(np.sum(np.abs(mlp_input[:128] - fake_embedding))), 0.0, tol=1e-4,
                 msg="embedding passthrough")


# =========================================================================
# Test: Action Mapping
# =========================================================================
def test_action_mapping():
    print("\n[TEST] Action Mapping")

    assert_equal(get_recommended_action(0.10), "all_clear", msg="risk=0.10")
    assert_equal(get_recommended_action(0.24), "all_clear", msg="risk=0.24")
    assert_equal(get_recommended_action(0.25), "stay_alert", msg="risk=0.25")
    assert_equal(get_recommended_action(0.49), "stay_alert", msg="risk=0.49")
    assert_equal(get_recommended_action(0.50), "reduce_speed", msg="risk=0.50")
    assert_equal(get_recommended_action(0.69), "reduce_speed", msg="risk=0.69")
    assert_equal(get_recommended_action(0.70), "slow_down", msg="risk=0.70")
    assert_equal(get_recommended_action(0.84), "slow_down", msg="risk=0.84")
    assert_equal(get_recommended_action(0.85), "brake_and_record", msg="risk=0.85")
    assert_equal(get_recommended_action(1.00), "brake_and_record", msg="risk=1.00")


# =========================================================================
# Run All Tests
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Node 3 — Phase 1 Data Handshake Unit Tests")
    print("=" * 60)

    test_speed_normalization()
    test_gps_normalization()
    test_cyclical_time()
    test_v_rel_normalization()
    test_telemetry_vector()
    test_label_encoding()
    test_detection_padding()
    test_anomaly_vector()
    test_mlp_assembly()
    test_action_mapping()

    print("\n" + "=" * 60)
    print(f"  Results: {_passed} passed, {_failed} failed")
    print("=" * 60)

    if _failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
        sys.exit(0)
