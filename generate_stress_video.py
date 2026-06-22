import cv2
import numpy as np
import os

def create_stress_video():
    input_path = "assets/video/standard_drive.mp4"
    output_path = "assets/video/stress_test.mp4"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0

    start_corruption = total_frames // 4
    end_corruption = (3 * total_frames) // 4

    print("Processing video frames and applying anomaly distortions...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_corruption <= frame_count <= end_corruption:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, 110)
            hsv_modified = cv2.merge((h, s, v))
            frame = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
            
            gauss_noise = np.random.normal(0, 35, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, gauss_noise)
            
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    create_stress_video()