import cv2
import onnxruntime as ort
import numpy as np
import time
import os
import zmq
import json
import csv

def calculate_anomaly_metrics(frame):
    small_frame = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    return blur_score, brightness_score

def run_inference_loop():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_path = os.path.join(base_dir, "C-RET_Engine", "assets", "models", "yolov8n.onnx")
    video_path = os.path.join(base_dir, "C-RET_Engine", "assets", "video", "stress_test1.mp4")
    csv_path = os.path.join(base_dir, "assets", "can_telemetry.csv")
    print(f"[Node 1] Resolving base path: {base_dir}")
    print(f"[Node 1] Using model path: {model_path}")
    print(f"[Node 1] Using video path: {video_path}")
    print(f"[Node 1] Using telemetry path: {csv_path}")
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[Node 1] Error: Missing model or video files.")
        return
    csv_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                csv_rows.append((float(r[0]), r[2], float(r[3])))
        print(f"[Node 1] Loaded {len(csv_rows)} CAN telemetry logs.")
    else:
        print("[Node 1] Warning: telemetry CSV not found. Running with baseline values.")
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5555")
    print("[Node 1] ZMQ publisher bound to tcp://127.0.0.1:5555")
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    frame_count = 0
    csv_idx = 0
    current_speed = 80.0
    COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    print("[Node 1] Starting video capture and inference loop...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        elapsed = frame_count / video_fps
        while csv_idx < len(csv_rows) and csv_rows[csv_idx][0] <= elapsed:
            ts, signal_name, val = csv_rows[csv_idx]
            if signal_name == "speed_kmph":
                current_speed = val
            csv_idx += 1
        blur, brightness = calculate_anomaly_metrics(frame)
        anomaly_flags = {
            "camera_blinded": bool(brightness > 180),
            "blur_detected": bool(blur < 30)
        }
        h, w, c = frame.shape
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        outputs = session.run(None, {input_name: img})
        output = outputs[0][0].T
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences > 0.5
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        detections = []
        if len(boxes) > 0:
            scale_x = w / 640.0
            scale_y = h / 640.0
            x1 = (boxes[:, 0] - boxes[:, 2] / 2.0) * scale_x
            y1 = (boxes[:, 1] - boxes[:, 3] / 2.0) * scale_y
            x2 = (boxes[:, 0] + boxes[:, 2] / 2.0) * scale_x
            y2 = (boxes[:, 1] + boxes[:, 3] / 2.0) * scale_y
            bboxes_cv = [[int(x), int(y), int(x2[i] - x), int(y2[i] - y)] for i, (x, y) in enumerate(zip(x1, y1))]
            indices = cv2.dnn.NMSBoxes(
                bboxes=bboxes_cv,
                scores=[float(c) for c in confidences],
                score_threshold=0.5,
                nms_threshold=0.4
            )
            if len(indices) > 0:
                for idx in indices.flatten():
                    detections.append({
                        "label": COCO_CLASSES[class_ids[idx]] if class_ids[idx] < len(COCO_CLASSES) else "unknown",
                        "confidence": float(confidences[idx]),
                        "box": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])]
                    })
        frame_data = {
            "timestamp": time.time(),
            "frame_shape": [h, w, c],
            "sensors": {
                "speed": current_speed,
                "lat": 28.6 + elapsed * 0.0001,
                "lon": 77.2 + elapsed * 0.0001
            },
            "detections": detections,
            "anomaly_flags": anomaly_flags
        }
        socket.send_multipart([
            json.dumps(frame_data).encode("utf-8"),
            frame.tobytes()
        ])
        print(f"[Node 1] Frame: {frame_count} | Glare: {anomaly_flags['camera_blinded']} | Objects: {len(detections)} | ZMQ Published")
        for det in detections:
            bx = det["box"]
            lbl = det["label"]
            cf = det["confidence"]
            cv2.rectangle(frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{lbl} {cf:.2f}", (int(bx[0]), int(bx[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {current_speed:.1f} km/h", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Node 1: YOLO Publisher", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("[Node 1] Video playback finished. Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    socket.close()
    context.term()

if __name__ == "__main__":
    run_inference_loop()