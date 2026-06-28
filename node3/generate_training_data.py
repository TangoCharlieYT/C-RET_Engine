"""
Node 3 — Phase 2: Synthetic Training Data Generator

Generates ~10,000 rows of simulated dashcam telemetry + perception data
across 6 driving scenarios, each with characteristic parameter distributions.

Scenarios:
    1. Daytime Highway Cruising   (low risk,   ~1,700 rows)
    2. Daytime City Driving       (low risk,   ~1,700 rows)
    3. Nighttime Highway          (med risk,   ~1,700 rows)
    4. Rain / Low Visibility      (med risk,   ~1,700 rows)
    5. Night + Glare + Erratic    (high risk,  ~1,600 rows)
    6. Imminent Crash             (high risk,  ~1,600 rows)

Output:
    data/synthetic_training_data.csv  (19 columns × ~10,000 rows)

Columns:
    Telemetry (6):   speed_norm, lat_norm, lon_norm, time_sin, time_cos, v_rel_norm
    Detections (5):  conf_1, conf_2, conf_3, conf_4, conf_5
    Top label (4):   label_car, label_person, label_truck, label_other
    Anomaly (3):     low_conf_flag, glare_flag, crash_prob
    Target (1):      risk_label

Usage:
    python node3/generate_training_data.py
"""

import os
import sys
import csv
import math
import random
import numpy as np

# Ensure project root is on the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.normalize import (
    normalize_speed,
    normalize_latitude,
    normalize_longitude,
    encode_time_cyclical,
    normalize_v_rel,
    encode_label_onehot,
    SPEED_MAX, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, V_REL_MAX,
    MAX_DETECTIONS, NUM_LABEL_CLASSES, LABEL_VOCAB,
)

# =========================================================================
# Constants
# =========================================================================

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_training_data.csv")
RANDOM_SEED = 42

CSV_COLUMNS = [
    # Telemetry (6)
    "speed_norm", "lat_norm", "lon_norm", "time_sin", "time_cos", "v_rel_norm",
    # Detections (5)
    "conf_1", "conf_2", "conf_3", "conf_4", "conf_5",
    # Top label one-hot (4)
    "label_car", "label_person", "label_truck", "label_other",
    # Anomaly flags (3)
    "low_conf_flag", "glare_flag", "crash_prob",
    # Target (1)
    "risk_label",
]

# Label options with approximate real-world frequency weights per scenario
LABELS = ["car", "person", "truck", "other"]


# =========================================================================
# Scenario Definitions
# =========================================================================

SCENARIOS = [
    {
        "name": "Daytime Highway Cruising",
        "count": 1700,
        "speed_range": (80, 120),
        "time_range_hours": (10, 16),       # 10:00 AM – 4:00 PM
        "glare_prob": 0.02,                 # Very rare glare
        "low_conf_prob": 0.03,              # Very rare low confidence
        "conf_range": (0.70, 0.95),
        "v_rel_range": (0.0, 5.0),
        "crash_prob_range": (0.0, 0.05),
        "risk_range": (0.05, 0.15),
        "label_weights": [0.60, 0.05, 0.30, 0.05],  # Mostly cars/trucks on highway
        "num_dets_range": (2, 5),           # Usually multiple vehicles
    },
    {
        "name": "Daytime City Driving",
        "count": 1700,
        "speed_range": (20, 60),
        "time_range_hours": (8, 18),        # 8:00 AM – 6:00 PM
        "glare_prob": 0.05,
        "low_conf_prob": 0.05,
        "conf_range": (0.60, 0.90),
        "v_rel_range": (0.0, 8.0),
        "crash_prob_range": (0.0, 0.10),
        "risk_range": (0.10, 0.25),
        "label_weights": [0.40, 0.35, 0.10, 0.15],  # Lots of pedestrians in city
        "num_dets_range": (1, 5),
    },
    {
        "name": "Nighttime Highway",
        "count": 1700,
        "speed_range": (60, 100),
        "time_range_hours": (22, 28),       # 10:00 PM – 4:00 AM (wraps past midnight)
        "glare_prob": 0.10,                 # Occasional headlight glare
        "low_conf_prob": 0.20,              # Darkness degrades detection
        "conf_range": (0.40, 0.70),
        "v_rel_range": (0.0, 10.0),
        "crash_prob_range": (0.02, 0.15),
        "risk_range": (0.25, 0.45),
        "label_weights": [0.55, 0.05, 0.30, 0.10],
        "num_dets_range": (0, 4),           # Fewer visible objects at night
    },
    {
        "name": "Rain / Low Visibility",
        "count": 1700,
        "speed_range": (30, 80),
        "time_range_hours": (0, 24),        # Any time of day
        "glare_prob": 0.08,                 # Water on lens causes some glare
        "low_conf_prob": 0.35,              # Rain significantly degrades detection
        "conf_range": (0.30, 0.60),
        "v_rel_range": (2.0, 12.0),
        "crash_prob_range": (0.05, 0.25),
        "risk_range": (0.35, 0.55),
        "label_weights": [0.50, 0.15, 0.20, 0.15],
        "num_dets_range": (0, 4),
    },
    {
        "name": "Nighttime + Glare + Erratic",
        "count": 1600,
        "speed_range": (80, 140),
        "time_range_hours": (22, 27),       # 10:00 PM – 3:00 AM
        "glare_prob": 0.80,                 # High glare (camera blinding)
        "low_conf_prob": 0.60,              # Severely compromised detections
        "conf_range": (0.15, 0.40),
        "v_rel_range": (10.0, 30.0),
        "crash_prob_range": (0.15, 0.50),
        "risk_range": (0.60, 0.85),
        "label_weights": [0.45, 0.10, 0.25, 0.20],
        "num_dets_range": (0, 3),           # Very few reliable detections
    },
    {
        "name": "Imminent Crash Scenario",
        "count": 1600,
        "speed_range": (60, 150),
        "time_range_hours": (0, 24),        # Any time
        "glare_prob": 0.40,                 # Sometimes glare present
        "low_conf_prob": 0.50,              # Often degraded
        "conf_range": (0.10, 0.50),
        "v_rel_range": (20.0, 50.0),        # Very high closing speed
        "crash_prob_range": (0.60, 1.0),    # Node 2 strongly signals crash
        "risk_range": (0.85, 1.0),
        "label_weights": [0.50, 0.20, 0.20, 0.10],
        "num_dets_range": (1, 5),           # Crash means something is in view
    },
]


# =========================================================================
# Helper: Sample time as UNIX-like timestamp (seconds since midnight)
# =========================================================================

def sample_time_seconds(hour_start: float, hour_end: float) -> float:
    """
    Sample a random time of day between hour_start and hour_end.
    Handles overnight wraparound (e.g., 22 to 28 means 10 PM to 4 AM).

    Returns:
        Seconds since midnight [0, 86400).
    """
    hour = random.uniform(hour_start, hour_end)
    # Wrap to 0–24 range for overnight scenarios
    hour = hour % 24.0
    return hour * 3600.0


# =========================================================================
# Helper: Generate a single sample row
# =========================================================================

def generate_sample(scenario: dict) -> dict:
    """
    Generate a single training sample from a scenario profile.

    Returns:
        dict with all 19 column values.
    """
    # --- Telemetry ---
    speed = random.uniform(*scenario["speed_range"])
    # GPS jittered within the Delhi region
    lat = random.uniform(LAT_MIN + 0.1, LAT_MAX - 0.1)
    lon = random.uniform(LON_MIN + 0.1, LON_MAX - 0.1)
    time_sec = sample_time_seconds(*scenario["time_range_hours"])
    v_rel = random.uniform(*scenario["v_rel_range"])

    # Normalize
    speed_norm = normalize_speed(speed)
    lat_norm = normalize_latitude(lat)
    lon_norm = normalize_longitude(lon)
    time_sin, time_cos = encode_time_cyclical(time_sec)
    v_rel_norm = normalize_v_rel(v_rel)

    # --- YOLO Detections ---
    num_dets = random.randint(*scenario["num_dets_range"])
    conf_min, conf_max = scenario["conf_range"]

    # Generate confidences sorted descending
    if num_dets > 0:
        confs = sorted(
            [random.uniform(conf_min, conf_max) for _ in range(num_dets)],
            reverse=True
        )
    else:
        confs = []

    # Pad/truncate to MAX_DETECTIONS (5)
    padded_confs = [0.0] * MAX_DETECTIONS
    for i in range(min(len(confs), MAX_DETECTIONS)):
        padded_confs[i] = round(confs[i], 4)

    # Top label one-hot (weighted random for the top detection)
    if num_dets > 0:
        top_label = random.choices(LABELS, weights=scenario["label_weights"], k=1)[0]
        label_onehot = encode_label_onehot(top_label).tolist()
    else:
        label_onehot = [0.0, 0.0, 0.0, 0.0]

    # --- Anomaly Flags ---
    low_conf_flag = 1.0 if random.random() < scenario["low_conf_prob"] else 0.0
    glare_flag = 1.0 if random.random() < scenario["glare_prob"] else 0.0
    crash_prob = round(random.uniform(*scenario["crash_prob_range"]), 4)

    # --- Risk Label ---
    risk_label = round(random.uniform(*scenario["risk_range"]), 4)

    # Correlation adjustments: make risk more realistic without bleeding into
    # the next scenario's bucket. Adjustments are scaled DOWN as the base
    # scenario risk rises, so low/medium scenarios can't get bumped into the
    # high-risk brake_and_record bucket.
    v_rel_factor = v_rel_norm * 0.04
    crash_factor = crash_prob * 0.04
    avg_conf = np.mean(padded_confs) if num_dets > 0 else 0.0
    conf_penalty = (1.0 - avg_conf) * 0.02
    flag_bump = glare_flag * 0.015 + low_conf_flag * 0.015

    # Suppress adjustments inside the imminent-crash scenario so its risk
    # ceiling stays near 1.0 without overshooting.
    is_critical = scenario["risk_range"][0] >= 0.85
    if is_critical:
        v_rel_factor *= 0.25
        crash_factor *= 0.25
        conf_penalty *= 0.25
        flag_bump *= 0.25

    risk_label = np.clip(
        risk_label + v_rel_factor + crash_factor + conf_penalty + flag_bump,
        0.0, 1.0
    )
    risk_label = round(float(risk_label), 4)

    return {
        "speed_norm": round(float(speed_norm), 6),
        "lat_norm": round(float(lat_norm), 6),
        "lon_norm": round(float(lon_norm), 6),
        "time_sin": round(float(time_sin), 6),
        "time_cos": round(float(time_cos), 6),
        "v_rel_norm": round(float(v_rel_norm), 6),
        "conf_1": padded_confs[0],
        "conf_2": padded_confs[1],
        "conf_3": padded_confs[2],
        "conf_4": padded_confs[3],
        "conf_5": padded_confs[4],
        "label_car": label_onehot[0],
        "label_person": label_onehot[1],
        "label_truck": label_onehot[2],
        "label_other": label_onehot[3],
        "low_conf_flag": low_conf_flag,
        "glare_flag": glare_flag,
        "crash_prob": crash_prob,
        "risk_label": risk_label,
    }


# =========================================================================
# Main: Generate Dataset
# =========================================================================

def generate_dataset():
    """Generate the full synthetic training dataset and save to CSV."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_rows = 0
    all_rows = []
    scenario_rows = []  # (scenario, row) tuples preserved pre-shuffle for stats

    print("=" * 60)
    print("  Node 3 — Synthetic Training Data Generator")
    print("=" * 60)

    for scenario in SCENARIOS:
        count = scenario["count"]
        name = scenario["name"]
        risk_lo, risk_hi = scenario["risk_range"]

        print(f"\n  Generating {count:,} samples: {name}")
        print(f"    Risk range: [{risk_lo:.2f}, {risk_hi:.2f}]")

        for _ in range(count):
            row = generate_sample(scenario)
            all_rows.append(row)
            scenario_rows.append((scenario, row))  # tagged before shuffle

        total_rows += count

    # Shuffle all rows to mix scenarios (prevents training bias from ordering)
    random.shuffle(all_rows)

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Summary Statistics ---
    risk_labels = [r["risk_label"] for r in all_rows]
    glare_count = sum(1 for r in all_rows if r["glare_flag"] == 1.0)
    low_conf_count = sum(1 for r in all_rows if r["low_conf_flag"] == 1.0)
    zero_det_count = sum(1 for r in all_rows if r["conf_1"] == 0.0)

    print(f"\n{'=' * 60}")
    print(f"  Dataset Generated Successfully!")
    print(f"{'=' * 60}")
    print(f"  Output file:       {OUTPUT_FILE}")
    print(f"  Total rows:        {total_rows:,}")
    print(f"  Columns:           {len(CSV_COLUMNS)}")
    print(f"  Risk label range:  [{min(risk_labels):.4f}, {max(risk_labels):.4f}]")
    print(f"  Risk label mean:   {np.mean(risk_labels):.4f}")
    print(f"  Glare flagged:     {glare_count:,} ({100.0 * glare_count / total_rows:.1f}%)")
    print(f"  Low-conf flagged:  {low_conf_count:,} ({100.0 * low_conf_count / total_rows:.1f}%)")
    print(f"  Zero-detection:    {zero_det_count:,} ({100.0 * zero_det_count / total_rows:.1f}%)")
    print(f"{'=' * 60}")

    # --- Per-Scenario Risk Distribution ---
    # Compute real stats from pre-shuffle tagged rows
    print(f"\n  Per-Scenario Risk Distribution:")
    print(f"  {'Scenario':<35} {'Count':>6}  {'Mean Risk':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-' * 75}")

    # Group risk labels by scenario (preserved pre-shuffle)
    from collections import defaultdict
    per_scenario_risks = defaultdict(list)
    for scenario, row in scenario_rows:
        per_scenario_risks[scenario["name"]].append(row["risk_label"])

    for scenario in SCENARIOS:
        name = scenario["name"]
        risks = per_scenario_risks[name]
        mean_r = float(np.mean(risks))
        min_r = float(np.min(risks))
        max_r = float(np.max(risks))
        print(f"  {name:<35} {len(risks):>6}  {mean_r:>10.4f}  {min_r:>8.4f}  {max_r:>8.4f}")

    # --- Per-Bucket Counts (action mapping thresholds) ---
    bucket_edges = [0.25, 0.50, 0.70, 0.85]
    bucket_names = ['0.00-0.25 (all_clear)', '0.25-0.50 (stay_alert)',
                    '0.50-0.70 (reduce_speed)', '0.70-0.85 (slow_down)',
                    '0.85-1.00 (brake_and_record)']
    bucket_counts = [0] * 5
    for v in [r["risk_label"] for r in all_rows]:
        for i, edge in enumerate(bucket_edges):
            if v < edge:
                bucket_counts[i] += 1
                break
        else:
            bucket_counts[4] += 1
    print(f"\n  Risk Buckets (action mapping):")
    for name, count in zip(bucket_names, bucket_counts):
        pct = 100.0 * count / total_rows
        print(f"    {name:<28} {count:>5} ({pct:>5.1f}%)")
    print()


if __name__ == "__main__":
    generate_dataset()
