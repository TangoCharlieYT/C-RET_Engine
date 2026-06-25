import os
import csv
import time
import threading
import numpy as np
from sklearn.ensemble import IsolationForest

class UEBAEngine:
    def __init__(self, csv_path, pipeline_state):
        self.csv_path = csv_path
        self.pipeline_state = pipeline_state
        self.deltas = {}
        self.last_timestamps = {}
        self.clf = IsolationForest(contamination=0.05, random_state=42)
        self.training_data = []
        self.trained = False

    def run_engine(self, stop_event):
        print(f"[Node 4: UEBA] Starting CAN telemetry simulation thread using CSV: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            print(f"[Node 4: UEBA] Error: CAN telemetry CSV file not found at {self.csv_path}!")
            return
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                rows.append(row)
        if not rows:
            print("[Node 4: UEBA] Warning: CAN telemetry CSV is empty!")
            return
        print(f"[Node 4: UEBA] Loaded {len(rows)} CAN messages. Starting real-time replay simulator...")
        start_sim_time = time.time()
        first_row_ts = float(rows[0][0])
        for row in rows:
            if stop_event.is_set():
                break
            ts = float(row[0])
            can_id = row[1]
            signal_name = row[2]
            value = float(row[3])
            sim_elapsed = time.time() - start_sim_time
            csv_elapsed = ts - first_row_ts
            delay = csv_elapsed - sim_elapsed
            if delay > 0:
                time.sleep(delay)
            self._process_message(ts, signal_name, value)
        print("[Node 4: UEBA] CAN telemetry simulation ended.")

    def _process_message(self, ts, signal_name, value):
        if signal_name not in self.deltas:
            self.deltas[signal_name] = []
        last_ts = self.last_timestamps.get(signal_name)
        self.last_timestamps[signal_name] = ts
        if last_ts is not None:
            delta = ts - last_ts
            self.deltas[signal_name].append(delta)
            if len(self.deltas[signal_name]) > 50:
                self.deltas[signal_name].pop(0)
            if signal_name == 'steering_angle_deg':
                if not self.trained:
                    self.training_data.append([value, delta])
                    if len(self.training_data) >= 100:
                        self.clf.fit(self.training_data)
                        self.trained = True
                        print("[Node 4: UEBA] Trained Isolation Forest baseline using first 100 steering messages.")
                else:
                    pred = self.clf.predict([[value, delta]])
                    if pred[0] == -1:
                        self._trigger_alert("Isolation Forest anomaly")
                if len(self.deltas[signal_name]) >= 10:
                    history = self.deltas[signal_name][:-1]
                    mean = np.mean(history)
                    std = np.std(history)
                    if std > 0:
                        z = (delta - mean) / std
                        if z < -3.0:
                            self._trigger_alert(f"Z-score anomaly (z={z:.2f})")

    def _trigger_alert(self, reason):
        if not self.pipeline_state.get_snapshot()["ueba_alert"]:
            print(f"[Node 4: UEBA] ALARM! CAN bus threat detected: {reason}. Pausing OTA updates and triggering lock.")
        self.pipeline_state.set_ueba_alert(True)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        tmp_dir = os.path.join(base_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        lock_file = os.path.join(tmp_dir, "ota_pause.lock")
        with open(lock_file, "w") as f:
            f.write(reason)
