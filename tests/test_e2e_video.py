"""E2E pipeline test over synthetic video frames."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.main_inference import _extract_top_detections
from node3.risk_engine import RiskEngine

engine = RiskEngine()


def make_frame(brightness=100, blur_strength=10, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(max(0, brightness - 30), min(255, brightness + 30),
                       size=(360, 640, 3), dtype=np.uint8)
    if blur_strength > 0:
        img = (img // blur_strength) * blur_strength
    return img


def yolo_like_output(image_mean_brightness):
    out = np.random.randn(1, 84, 8400).astype(np.float32) * 0.1
    base = 0.3 if image_mean_brightness < 150 else 0.7
    if image_mean_brightness > 200:
        base = 0.95
    out[0, 4:, :50] = np.random.uniform(base - 0.05, base + 0.05, size=(80, 50))
    return out


print("=== Pipeline: simulate stress video ===")
print("%5s %10s %8s %-20s" % ("frame", "brightness", "risk", "action"))
scores = []
actions = []
for i, brightness in enumerate([80, 100, 120, 180, 210, 240, 200, 150, 100, 80]):
    frame = make_frame(brightness=brightness)
    yolo_out = yolo_like_output(brightness)
    dets = _extract_top_detections(yolo_out)
    proxy_crash = float(np.clip((brightness - 150) / 50, 0, 1))
    fd = {
        "sensors": {"speed": 60, "lat": 28.6, "lon": 77.2},
        "timestamp": 43200.0 + i * 30,
        "detections": dets,
    }
    ts = {
        "anomaly_flags": {
            "low_conf": brightness < 80,
            "glare": brightness > 180,
        },
        "kinematics": {
            "v_rel": max(0, (brightness - 100) / 5),
            "crash_prob": proxy_crash,
        },
    }
    out = engine.infer(fd, ts)
    scores.append(out["risk_score"])
    actions.append(out["recommended_action"])
    print("%5d %10.1f %8.3f %-20s" % (
        i, brightness, out["risk_score"], out["recommended_action"]
    ))

print()
print("Risk peaks during degradation window?",
      scores[3] <= scores[5] and scores[5] >= 0.7)
print("Risk recedes back to low after recovery?",
      scores[-1] < scores[3] and scores[-1] < 0.5)
print("brake_and_record triggered at peak?", "brake_and_record" in actions[3:6])