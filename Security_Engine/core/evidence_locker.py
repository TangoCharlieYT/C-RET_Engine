import os
import subprocess
import json
import time
import shutil

class EvidenceLocker:
    def __init__(self, ring_buffer, pipeline_state, crypto_ledger, output_dir):
        self.ring_buffer = ring_buffer
        self.pipeline_state = pipeline_state
        self.crypto_ledger = crypto_ledger
        self.output_dir = os.path.abspath(output_dir)
        self.warning_active = False
        self.critical_active = False
        self.post_crash_counter = 0
        self.event_ts = 0.0
        self.dat_path = ""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Node 4: EvidenceLocker] Initialized. Output directory: {self.output_dir}")

    def trigger_warning(self, timestamp):
        if self.warning_active or self.critical_active:
            return
        self.warning_active = True
        self.event_ts = timestamp
        self.pipeline_state.set_active_event_ts(timestamp)
        self.dat_path = os.path.join(self.output_dir, f"tmp_evt_{int(timestamp)}.dat")
        print(f"[Node 4: EvidenceLocker] WARNING state pre-emptive disk cache initiated at timestamp {timestamp}")
        snapshot = self.ring_buffer.snapshot()
        print(f"[Node 4: EvidenceLocker] Writing warning history snapshot of {len(snapshot)} frames to: {self.dat_path}")
        with open(self.dat_path, "wb") as f:
            for jpeg_bytes in snapshot:
                f.write(jpeg_bytes)

    def trigger_critical(self, timestamp):
        if self.critical_active:
            return
        self.critical_active = True
        self.ring_buffer.lock_buffer()
        if not self.warning_active:
            self.event_ts = timestamp
            self.pipeline_state.set_active_event_ts(timestamp)
            self.dat_path = os.path.join(self.output_dir, f"tmp_evt_{int(timestamp)}.dat")
            print(f"[Node 4: EvidenceLocker] Direct CRITICAL lock initiated at timestamp {timestamp}")
            snapshot = self.ring_buffer.snapshot()
            print(f"[Node 4: EvidenceLocker] Writing critical history snapshot of {len(snapshot)} frames to: {self.dat_path}")
            with open(self.dat_path, "wb") as f:
                for jpeg_bytes in snapshot:
                    f.write(jpeg_bytes)
        else:
            print(f"[Node 4: EvidenceLocker] Upgraded WARNING state to CRITICAL lock at timestamp {timestamp}")
        self.pipeline_state.set_locked(True)
        self.post_crash_counter = 0

    def handle_frame(self, jpeg_bytes):
        if self.critical_active:
            if jpeg_bytes:
                with open(self.dat_path, "ab") as f:
                    f.write(jpeg_bytes)
                self.post_crash_counter += 1
                if self.post_crash_counter % 30 == 0:
                    print(f"[Node 4: EvidenceLocker] Appended post-crash frame: {self.post_crash_counter}/150")
            if self.post_crash_counter >= 150:
                print(f"[Node 4: EvidenceLocker] Captured all 150 post-crash frames. Compiling MP4 clip...")
                self._finalize_event()

    def _finalize_event(self):
        ts_int = int(self.event_ts)
        mp4_filename = f"evt_{ts_int}.mp4"
        mp4_path = os.path.join(self.output_dir, mp4_filename)
        ffmpeg_bin = "ffmpeg"
        if not shutil.which("ffmpeg"):
            win_user = os.path.expanduser('~')
            winget_path = os.path.join(win_user, "AppData", "Local", "Microsoft", "WinGet", "Packages")
            if os.path.exists(winget_path):
                for root, dirs, files in os.walk(winget_path):
                    if "ffmpeg.exe" in files:
                        ffmpeg_bin = os.path.join(root, "ffmpeg.exe")
                        break
        print(f"[Node 4: EvidenceLocker] Using FFmpeg path: {ffmpeg_bin}")
        cmd = [
            ffmpeg_bin, "-y",
            "-r", "30",
            "-f", "mjpeg",
            "-i", self.dat_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            mp4_path
        ]
        try:
            print(f"[Node 4: EvidenceLocker] Running compilation command: {' '.join(cmd)}")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"[Node 4: EvidenceLocker] MP4 clip successfully compiled at {mp4_path}")
        except Exception as e:
            print(f"[Node 4: EvidenceLocker] Error: FFmpeg compilation failed: {str(e)}")
        if os.path.exists(self.dat_path):
            try:
                os.remove(self.dat_path)
                print(f"[Node 4: EvidenceLocker] Deleted temporary file: {self.dat_path}")
            except Exception as e:
                print(f"[Node 4: EvidenceLocker] Warning: Could not delete temporary dat file: {str(e)}")
        snap = self.pipeline_state.get_snapshot()
        event_bundle = {
            "event_id": f"evt_{ts_int}",
            "timestamp": self.event_ts,
            "vehicle_id": "CRET-EDGE-01",
            "pipeline_version": "1.0.0",
            "frame_data": snap["frame_data"],
            "triage_state": snap["triage"],
            "risk_state": snap["risk"],
            "ueba_alert": snap["ueba_alert"],
            "media": {
                "mp4_filename": mp4_filename,
                "frame_count": 450,
                "duration_seconds": 15.0
            }
        }
        try:
            event_bundle = self.crypto_ledger.sign_payload(mp4_path, event_bundle)
        except Exception as e:
            print(f"[Node 4: EvidenceLocker] Warning: HMAC signing failed: {str(e)}")
            if "security" not in event_bundle:
                event_bundle["security"] = {"hmac_signature": "UNSIGNED", "signing_algorithm": "HMAC-SHA256"}
        bundle_path = os.path.join(self.output_dir, f"event_bundle_{ts_int}.json")
        with open(bundle_path, "w") as f:
            json.dump(event_bundle, f, indent=4)
        print(f"[Node 4: EvidenceLocker] Saved JSON metadata bundle to {bundle_path}")
        self.crypto_ledger.queue_for_sync(event_bundle, mp4_path)
        self.ring_buffer.unlock_buffer()
        self.pipeline_state.set_locked(False)
        self.pipeline_state.set_active_event_ts(0.0)
        self.warning_active = False
        self.critical_active = False
        self.post_crash_counter = 0
