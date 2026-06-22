# C-RET Engine: Cognitive Real-time Edge Triage

## Overview
C-RET Engine is a highly optimized, hardware-accelerated edge AI pipeline designed to process live dashcam feeds. It dynamically detects sensor degradation (e.g., camera blinding, dirt, adversarial noise) and executes data triage protocols to save bandwidth while maintaining critical safety operations.

## Pillar 1: Vision Inference & Mathematical Triage
* **Hardware Acceleration:** Utilizes DirectML to run YOLOv8 ONNX graphs directly on constrained edge GPUs.
* **Anomaly Detection:** Employs Sub-Resolution Profiling (Laplacian Variance and Mean Intensity) to detect structural and lighting failures in real-time.
* **Smart Triage:** Deletes "Green State" (safe) frames to save 99% bandwidth, while securely queuing "Amber State" (degraded) frames to a local outbox with a cooldown gate.

## Setup Instructions for Team Members
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\Activate.ps1`
4. Install core dependencies: `pip install onnxruntime-directml opencv-python numpy`
5. Place a 720p dashcam video named `stress_test.mp4` into the `assets/video/` folder.
6. Run the engine: `python core/main_inference.py`