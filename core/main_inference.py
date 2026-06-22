import cv2
import onnxruntime as ort
import numpy as np
import time
import os
import glob

def calculate_anomaly_metrics(frame):
    small_frame = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_score = np.mean(gray)
    return blur_score, brightness_score

def run_inference_loop():
    model_path = "assets/models/yolov8n.onnx"
    video_path = "assets/video/stress_test.mp4"
    queue_dir = "assets/cloud_queue"

    # Ensure our simulated cloud upload folder exists
    os.makedirs(queue_dir, exist_ok=True)
    
    # --- AUTO-CLEAN CACHE ---
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

    cap = cv2.VideoCapture(video_path)
    
    # --- TRIAGE METRICS ---
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
        
        # Anomaly Logic Gate
        if brightness > 180 or blur < 30:
            is_anomaly = True
            state_color = (0, 140, 255)
            state_text = "SYSTEM STATE: AMBER (DEGRADED)"
            
            # Save frame if cooldown has passed
            if (time.time() - last_upload_time) > upload_cooldown_sec:
                filename = os.path.join(queue_dir, f"anomaly_{int(time.time()*1000)}.jpg")
                cv2.imwrite(filename, frame)
                frames_uploaded += 1
                last_upload_time = time.time()
                
        # Calculate Bandwidth Saved
        bandwidth_saved = 100.0
        if total_frames_processed > 0:
            bandwidth_saved = (1.0 - (frames_uploaded / total_frames_processed)) * 100.0
            
        # ONNX Inference
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        outputs = session.run(None, {input_name: img})
        
        # Telemetry Display
        latency_ms = (time.time() - start_time) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        cv2.putText(frame, state_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Cloud Uploads: {frames_uploaded} | Bandwidth Saved: {bandwidth_saved:.2f}%", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        display_frame = cv2.resize(frame, (1024, 576))
        cv2.imshow("C-RET Engine: Triage Gate", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal Triage Report:")
    print(f"Total Frames Processed: {total_frames_processed}")
    print(f"Frames Sent to Cloud: {frames_uploaded}")
    print(f"Bandwidth Saved: {bandwidth_saved:.2f}%\n")

if __name__ == "__main__":
    run_inference_loop()