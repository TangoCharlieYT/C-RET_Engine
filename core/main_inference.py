import cv2
import onnxruntime as ort
import numpy as np
import time
import os
import glob
import sys

# Ensure project root is on the path so node3/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from node3.risk_engine import RiskEngine


def calculate_anomaly_metrics(frame):
    small_frame = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    return blur_score, brightness_score


def _extract_top_detections(yolo_output):
    """
    Best-effort placeholder: pull the top-5 confidence values from a raw
    YOLOv8 ONNX output tensor and synthesize stub labels for the
    RiskEngine input contract.

    A production implementation would run NMS + class decoding here.
    This is intentionally lightweight for the integration milestone.
    """
    try:
        if yolo_output.ndim == 3:
            preds = yolo_output[0]
            class_scores = preds[4:, :]
            top_scores = np.sort(class_scores.max(axis=0))[-5:][::-1]
            top_scores = np.clip(top_scores, 0.0, 1.0)
            coco_common = ["car", "person", "truck", "car", "person"]
            detections = []
            for i, score in enumerate(top_scores):
                if score < 0.05:
                    break
                detections.append({
                    "label": coco_common[i % len(coco_common)],
                    "confidence": float(score),
                    "box": [0, 0, 1, 1],
                })
            return detections
    except Exception:
        pass
    return []


def run_inference_loop():
    model_path = "assets/models/yolov8n.onnx"
    video_path = "assets/video/stress_test.mp4"
    queue_dir = "assets/cloud_queue"

    os.makedirs(queue_dir, exist_ok=True)

    print("Clearing previous triage cache...")
    old_files = glob.glob(os.path.join(queue_dir, "*.jpg"))
    for f in old_files:
        os.remove(f)

    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("Missing model or video files.")
        return

    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for _ in range(3):
        session.run(None, {input_name: dummy_input})

    # --- NODE 3: Risk Engine (Phase 5) ---
    print("Initializing Node 3 -- Risk Engine...")
    try:
        risk_engine = RiskEngine(
            encoder_path="node3/models/context_encoder.onnx",
            mlp_path="node3/models/safety_risk_mlp.onnx",
        )
        print("  RiskEngine ready (providers: %s)" % risk_engine.active_providers)
        risk_engine_available = True
    except FileNotFoundError as e:
        print("  [!] RiskEngine unavailable: %s" % e)
        print("  [!] Run node3/export_to_onnx.py first to produce ONNX models.")
        risk_engine_available = False

    cap = cv2.VideoCapture(video_path)

    total_frames_processed = 0
    frames_uploaded = 0
    last_upload_time = 0
    upload_cooldown_sec = 1.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        total_frames_processed += 1

        blur, brightness = calculate_anomaly_metrics(frame)

        is_anomaly = False
        state_color = (0, 255, 0)
        state_text = "SYSTEM STATE: GREEN (NOMINAL)"

        if brightness > 180 or blur < 30:
            is_anomaly = True
            state_color = (0, 140, 255)
            state_text = "SYSTEM STATE: AMBER (DEGRADED)"

            if (time.time() - last_upload_time) > upload_cooldown_sec:
                filename = os.path.join(queue_dir, "anomaly_%d.jpg" % int(time.time() * 1000))
                cv2.imwrite(filename, frame)
                frames_uploaded += 1
                last_upload_time = time.time()

        bandwidth_saved = 100.0
        if total_frames_processed > 0:
            bandwidth_saved = (1.0 - (frames_uploaded / total_frames_processed)) * 100.0

        # ONNX Inference
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        outputs = session.run(None, {input_name: img})

        # --- NODE 3: Risk Engine inference ---
        risk_state = None
        if risk_engine_available:
            detections = _extract_top_detections(outputs[0])
            proxy_crash_prob = float(np.clip((brightness - 150.0) / 50.0, 0.0, 1.0))

            frame_data = {
                "sensors": {
                    "speed": 60.0,
                    "lat":   28.6,
                    "lon":   77.2,
                },
                "timestamp":  time.time(),
                "detections": detections,
            }
            triage_state = {
                "anomaly_flags": {
                    "low_conf": brightness < 80 or blur < 20,
                    "glare":     brightness > 180,
                },
                "kinematics": {
                    "v_rel":      max(0.0, (brightness - 100.0) / 5.0),
                    "crash_prob": proxy_crash_prob,
                },
            }
            risk_state = risk_engine.infer(frame_data, triage_state)

        # Telemetry Display
        latency_ms = (time.time() - start_time) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        cv2.putText(frame, state_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, "FPS: %.1f | Latency: %.1f ms" % (fps, latency_ms), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Cloud Uploads: %d | Bandwidth Saved: %.2f%%" % (frames_uploaded, bandwidth_saved), (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if risk_state is not None:
            risk_pct = risk_state["risk_score"] * 100.0
            action = risk_state["recommended_action"]
            action_color = (0, 255, 0)
            if action == "stay_alert":
                action_color = (0, 200, 200)
            elif action in ("reduce_speed", "slow_down"):
                action_color = (0, 140, 255)
            elif action == "brake_and_record":
                action_color = (0, 0, 255)

            cv2.putText(frame, "RISK SCORE: %5.1f%%" % risk_pct, (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "ACTION: " + action, (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
            cv2.putText(frame, "Node 3: %.2f ms" % risk_engine.last_latency_ms,
                        (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        display_frame = cv2.resize(frame, (1024, 576))
        cv2.imshow("C-RET Engine: Triage Gate", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal Triage Report:")
    print("Total Frames Processed: %d" % total_frames_processed)
    print("Frames Sent to Cloud: %d" % frames_uploaded)
    print("Bandwidth Saved: %.2f%%\n" % bandwidth_saved)


if __name__ == "__main__":
    run_inference_loop()