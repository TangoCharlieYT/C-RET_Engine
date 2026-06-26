"""
Node 3 -- Phase 5: Production Inference Engine

RiskEngine is the per-frame inference class that wraps both ONNX sessions
(encoder + MLP) and exposes the public output contract:

    {
        "context_embedding":  [0.12, -0.34, ...],   # 128-dim vector
        "risk_score":         0.87,                  # float in [0.0, 1.0]
        "recommended_action": "brake_and_record"     # str
    }

The engine wraps the Phase 1 normalization helpers (node3.normalize) and
the Phase 4 action mapper, so callers can pass raw FrameData + TriageState
directly.

Hardware acceleration: tries DmlExecutionProvider first, falls back to
CPUExecutionProvider, mirroring the YOLOv8 session in core/main_inference.py.
"""

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from node3.normalize import (
    build_telemetry_vector,
    pad_detections,
    build_anomaly_vector,
    get_recommended_action,
)


# =========================================================================
# Constants -- mirror the data contract from Plan §1.5
# =========================================================================
TELEMETRY_DIM = 6
DETECTION_DIM = 5
LABEL_DIM = 4
ANOMALY_DIM = 3
EMBED_DIM = 128
MLP_INPUT_DIM = 140  # 128 + 5 + 4 + 3


# =========================================================================
# RiskEngine
# =========================================================================

class RiskEngine:
    """
    Per-frame context encoder + risk MLP inference.

    Usage:
        engine = RiskEngine(
            encoder_path="node3/models/context_encoder.onnx",
            mlp_path="node3/models/safety_risk_mlp.onnx",
        )
        state = engine.infer(frame_data, triage_state)
        print(state["risk_score"], state["recommended_action"])

    Where:
        frame_data = {
            "sensors":     {"speed": 80.0, "lat": 28.6, "lon": 77.2},
            "timestamp":   1687372200.0,
            "detections":  [{"label": "car", "confidence": 0.92, "box": [...]}]
        }
        triage_state = {
            "anomaly_flags": {"low_conf": False, "glare": True},
            "kinematics":    {"v_rel": 12.5, "crash_prob": 0.18}
        }
    """

    def __init__(
        self,
        encoder_path: str = "node3/models/context_encoder.onnx",
        mlp_path: str = "node3/models/safety_risk_mlp.onnx",
        providers: List[str] = None,
    ):
        if providers is None:
            # Mirror YOLOv8 fallback chain in core/main_inference.py
            available = ort.get_available_providers()
            providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder ONNX not found: {encoder_path}")
        if not os.path.exists(mlp_path):
            raise FileNotFoundError(f"MLP ONNX not found: {mlp_path}")

        # Load both ONNX sessions (single encoder session is reused)
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=providers)
        self.mlp_sess = ort.InferenceSession(mlp_path, providers=providers)

        self.encoder_input = self.encoder_sess.get_inputs()[0].name
        self.mlp_input = self.mlp_sess.get_inputs()[0].name

        self.providers = providers
        self._last_latency_ms = 0.0

        # Warm-up runs (per Plan §5.2: same pattern as YOLOv8 in main_inference.py)
        self._warmup()

    # ------------------------------------------------------------------
    # Warm-up: prime the DirectML/CUDA kernel caches
    # ------------------------------------------------------------------
    def _warmup(self, n: int = 3):
        for _ in range(n):
            tel = np.zeros((1, TELEMETRY_DIM), dtype=np.float32)
            self.encoder_sess.run(None, {self.encoder_input: tel})
            mlp_in = np.zeros((1, MLP_INPUT_DIM), dtype=np.float32)
            self.mlp_sess.run(None, {self.mlp_input: mlp_in})

    # ------------------------------------------------------------------
    # Public API -- per the Plan §5.3 method list
    # ------------------------------------------------------------------

    def normalize_telemetry(self, sensors: Dict, timestamp: float, v_rel: float = 0.0) -> np.ndarray:
        """
        Build the 6-dim normalized telemetry vector.

        Returns:
            np.ndarray of shape (6,), dtype float32
        """
        return build_telemetry_vector(sensors, timestamp, v_rel).astype(np.float32)

    def pad_detections(self, detections: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize variable-length YOLO detections into fixed-size vectors.

        Returns:
            (confidences (5,), label_onehot (4,))
        """
        conf, label = pad_detections(detections)
        return conf.astype(np.float32), label.astype(np.float32)

    def encode_anomalies(self, triage_state: Dict) -> np.ndarray:
        """
        Encode Node 2's anomaly flags + crash_prob into a 3-dim vector.

        Returns:
            np.ndarray of shape (3,), dtype float32
        """
        return build_anomaly_vector(triage_state).astype(np.float32)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def infer(self, frame_data: Dict, triage_state: Dict) -> Dict:
        """
        Run the full Node 3 inference pipeline on one frame.

        Args:
            frame_data:   dict matching the FrameData contract
                          {"sensors": {...}, "timestamp": float, "detections": [...]}
            triage_state: dict matching the TriageState contract
                          {"anomaly_flags": {...}, "kinematics": {...}}

        Returns:
            {
                "context_embedding":  np.ndarray (128,),
                "risk_score":         float in [0.0, 1.0],
                "recommended_action": str
            }
        """
        t0 = time.perf_counter()

        # ----- Stage 1: normalize raw inputs -----
        sensors = frame_data.get("sensors", {}) or {}
        timestamp = float(frame_data.get("timestamp", 0.0))
        v_rel = float(
            triage_state.get("kinematics", {}).get("v_rel", 0.0)
        )
        tel_vec = self.normalize_telemetry(sensors, timestamp, v_rel)        # (6,)
        detections = frame_data.get("detections", []) or []
        conf_vec, label_vec = self.pad_detections(detections)                # (5,), (4,)
        anom_vec = self.encode_anomalies(triage_state)                       # (3,)

        # ----- Stage 2: ONNX encoder (6 -> 128) -----
        tel_batch = tel_vec.reshape(1, TELEMETRY_DIM).astype(np.float32)
        embedding = self.encoder_sess.run(
            None, {self.encoder_input: tel_batch}
        )[0].reshape(EMBED_DIM)

        # ----- Stage 3: ONNX MLP (140 -> 1) -----
        mlp_in = np.concatenate(
            [embedding, conf_vec, label_vec, anom_vec], axis=0
        ).reshape(1, MLP_INPUT_DIM).astype(np.float32)

        risk_score = float(
            self.mlp_sess.run(None, {self.mlp_input: mlp_in})[0].flatten()[0]
        )
        # Numerical safety: clamp to [0, 1]
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        action = get_recommended_action(risk_score)

        self._last_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "context_embedding":  embedding.astype(np.float32),
            "risk_score":         risk_score,
            "recommended_action": action,
        }

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    @property
    def last_latency_ms(self) -> float:
        """Wall-clock latency of the most recent `infer()` call (milliseconds)."""
        return self._last_latency_ms

    @property
    def active_providers(self) -> List[str]:
        return self.providers


# =========================================================================
# CLI smoke test (no video required): exercise the engine on synthetic data
# =========================================================================

def _self_test():
    """Sanity-check the RiskEngine on a few hand-crafted frames."""
    print("=" * 60)
    print("  RiskEngine -- self-test")
    print("=" * 60)

    engine = RiskEngine()

    cases = [
        # (label, frame_data, triage_state)
        (
            "Daytime Highway Cruise (low risk)",
            {"sensors": {"speed": 100.0, "lat": 28.6, "lon": 77.2},
             "timestamp": 43200.0,  # noon
             "detections": [{"label": "car", "confidence": 0.88, "box": [0, 0, 1, 1]}]},
            {"anomaly_flags": {"low_conf": False, "glare": False},
             "kinematics": {"v_rel": 1.0, "crash_prob": 0.01}},
        ),
        (
            "Nighttime + Glare (high risk)",
            {"sensors": {"speed": 110.0, "lat": 28.5, "lon": 77.0},
             "timestamp": 79200.0,  # 22:00
             "detections": [{"label": "car", "confidence": 0.20, "box": [0, 0, 1, 1]}]},
            {"anomaly_flags": {"low_conf": True, "glare": True},
             "kinematics": {"v_rel": 25.0, "crash_prob": 0.45}},
        ),
        (
            "Imminent Crash (max risk)",
            {"sensors": {"speed": 130.0, "lat": 28.7, "lon": 77.1},
             "timestamp": 36000.0,
             "detections": [{"label": "person", "confidence": 0.10, "box": [0, 0, 1, 1]}]},
            {"anomaly_flags": {"low_conf": True, "glare": False},
             "kinematics": {"v_rel": 45.0, "crash_prob": 0.95}},
        ),
    ]

    for label, fd, ts in cases:
        out = engine.infer(fd, ts)
        print(f"\n  {label}")
        print(f"    risk_score:         {out['risk_score']:.4f}")
        print(f"    recommended_action: {out['recommended_action']}")
        print(f"    latency:            {engine.last_latency_ms:.2f} ms")
        print(f"    embedding[:5]:      {out['context_embedding'][:5]}")

    print("\n" + "=" * 60)
    print("  Self-test complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _self_test()