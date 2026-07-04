# 🛡️ C-RET Edge Mesh: Local Integration MVP

Welcome to the **Contextual Resilient Edge Triage (C-RET)** local integration environment. This repository currently connects **Node 1 (Vision & Ingestion)** and **Node 4 (Security & Locker)**, while using mock publishers to simulate Nodes 2 and 3.

This setup proves our core edge architecture: running lightweight AI locally, keeping RAM footprints under 30MB, detecting cyber-physical threats (CAN bus flooding), and cryptographically locking evidence footage into an SQLite database without relying on the cloud.

## 🚀 How It Works (Current Flow)
* **Node 1 (Vision):** Ingests the raw dashcam video, runs the quantized YOLOv8n ONNX model, detects glare anomalies, and streams a JSON contract + raw frame bytes over ZMQ (`port 5555`).
* **Node 4 (Security Gatekeeper):** Listens to all nodes. It buffers frames efficiently (JPEG compression) to prevent OOM errors. It simultaneously monitors a simulated CAN bus for Z-score anomalies (UEBA). 
* **The Evidence Locker:** If Node 4 detects a CAN bus attack, or receives a high `crash_prob` / `risk_score` from the mock nodes, it instantly locks the memory buffer, uses FFmpeg to compile a 15-second pre/post-crash MP4, cryptographically signs it (HMAC-SHA256), and queues it in a local database for cloud sync.
* **Mock Nodes 2 & 3:** These publish simulated data on ZMQ `ports 5556` and `5557`. Because we use a plug-and-play pub/sub architecture, when the real Triage and Risk nodes are built, they will seamlessly replace these mocks with **zero code changes** required in Node 4.

---

## 📂 Core Directory Structure

```text
CRET/
├── assets/
│   ├── evidence_locker/       # Saved MP4s, JSON bundles, and lock files
│   ├── can_telemetry.csv      # Simulated CAN bus telemetry data
│   └── stress_test.mp4        # Input dashcam feed for Node 1
├── C-RET_Engine/
│   └── core/
│       └── main_inference.py  # NODE 1: YOLO Vision & ZMQ Streaming
├── Security_Engine/
│   ├── core/
│   │   ├── ring_buffer.py     # Memory Guardian (<30MB RAM overhead limit)
│   │   ├── ueba_engine.py     # CAN bus Isolation Forest anomaly detector
│   │   ├── evidence_locker.py # Dual-flash MP4 video compiler
│   │   └── crypto_ledger.py   # HMAC-SHA256 signing & SQLite queue sync
│   ├── tools/
│   │   └── mock_nodes.py      # MOCK: Simulates Node 2 (Triage) & Node 3 (Risk)
│   ├── config.env             # Local configuration and secret keys
│   ├── metadata.db            # Local SQLite ledger for cloud-sync payload queuing
│   └── run_node4.py           # NODE 4: Central Orchestrator & Gatekeeper
└── requirements.txt           # Python dependencies

```

*(Note: Temporary files, caches, virtual environments, and data-generation scripts are excluded from source control via `.gitignore`)*

---

## ⚙️ Installation & Setup

Before running the pipeline, ensure you have Python 3.9+ installed, as well as **FFmpeg** installed on your system PATH (required for compiling the MP4 videos).

**1. Create and activate the virtual environment:**

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

```

**2. Install dependencies:**

```powershell
pip install -r requirements.txt

```

---

## 🏃 Execution Instructions

To run the complete pipeline locally, you must open **three separate terminal windows** at the project root (`CRET/`). Ensure your virtual environment is activated in all three terminals.

**Terminal 1: Launch Mock Nodes (Triage & Risk)**
*Starts publishers on ports 5556 and 5557 that occasionally spike threat metrics to trigger the locker.*

```powershell
python Security_Engine/tools/mock_nodes.py

```

**Terminal 2: Launch Node 4 (Security & Locker Engine)**
*Starts the memory guardian, CAN bus anomaly monitor, and DB sync worker. It will wait for incoming data.*

```powershell
python Security_Engine/run_node4.py

```

**Terminal 3: Launch Node 1 (Vision Publisher)**
*Starts the video ingestion, YOLOv8 object detection, glare analysis, and frame streaming.*

```powershell
python C-RET_Engine/core/main_inference.py

```

### 📊 What to expect in the Terminal

Check **Terminal 2**. You will see clean console logs showing Node 4 connecting to the ZMQ sockets, tracking the frame buffers, detecting simulated CAN bus attacks (Z-score anomalies), triggering the Critical Evidence Locker, and successfully writing signed MP4s to the `assets/evidence_locker` directory!



