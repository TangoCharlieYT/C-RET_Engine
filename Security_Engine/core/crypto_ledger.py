import os
import hmac
import hashlib
import json
import time
import sqlite3
import threading
import requests

class CryptoLedger:
    def __init__(self, secret_key, db_path, appwrite_config=None):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.db_path = os.path.abspath(db_path)
        self.appwrite_config = appwrite_config or {}
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cloud_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    bundle_json TEXT,
                    mp4_path TEXT,
                    status TEXT DEFAULT 'PENDING',
                    retry_count INTEGER DEFAULT 0,
                    created_at REAL,
                    synced_at REAL
                )
            """)
            conn.commit()
        print(f"[Node 4: CryptoLedger] SQLite database initialized at {self.db_path}")

    def sign_payload(self, mp4_path, event_bundle):
        if not os.path.exists(mp4_path):
            print(f"[Node 4: CryptoLedger] Error: MP4 video file not found for signing: {mp4_path}")
            raise FileNotFoundError(f"MP4 file not found: {mp4_path}")
        with open(mp4_path, 'rb') as f:
            mp4_bytes = f.read()
        bundle_str = json.dumps(event_bundle, sort_keys=True)
        combined_payload = mp4_bytes + bundle_str.encode()
        signature = hmac.new(self.secret_key, combined_payload, hashlib.sha256).hexdigest()
        if "security" not in event_bundle:
            event_bundle["security"] = {}
        event_bundle["security"]["hmac_signature"] = signature
        event_bundle["security"]["signing_algorithm"] = "HMAC-SHA256"
        print(f"[Node 4: CryptoLedger] Cryptographically signed video + telemetry payload for event: {event_bundle.get('event_id')}")
        return event_bundle

    def queue_for_sync(self, event_bundle, mp4_path):
        event_id = event_bundle.get("event_id", f"evt_{int(time.time())}")
        bundle_json = json.dumps(event_bundle)
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO cloud_queue (event_id, bundle_json, mp4_path, status, created_at)
                    VALUES (?, ?, ?, 'PENDING', ?)
                """, (event_id, bundle_json, mp4_path, time.time()))
                conn.commit()
                print(f"[Node 4: CryptoLedger] Enqueued event {event_id} in local database queue.")
            except sqlite3.IntegrityError:
                print(f"[Node 4: CryptoLedger] Warning: Event {event_id} already exists in database queue.")

    def run_sync_worker(self, stop_event):
        print("[Node 4: CryptoLedger] SQLite Cloud Queue Sync Worker thread started.")
        while not stop_event.is_set():
            self._sync_pending_events()
            time.sleep(5)

    def _sync_pending_events(self):
        endpoint = self.appwrite_config.get("APPWRITE_ENDPOINT", "REPLACE_ME")
        project_id = self.appwrite_config.get("APPWRITE_PROJECT_ID", "REPLACE_ME")
        api_key = self.appwrite_config.get("APPWRITE_API_KEY", "REPLACE_ME")
        func_id = self.appwrite_config.get("APPWRITE_FUNCTION_ID", "REPLACE_ME")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cloud_queue WHERE status = 'PENDING'")
            rows = cursor.fetchall()
        for row in rows:
            event_id = row["event_id"]
            bundle_json = row["bundle_json"]
            mp4_path = row["mp4_path"]
            retry_count = row["retry_count"]
            if any(v == "REPLACE_ME" or not v for v in [endpoint, project_id, api_key, func_id]):
                with self.lock, sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE cloud_queue 
                        SET status = 'SYNCED', synced_at = ? 
                        WHERE event_id = ?
                    """, (time.time(), event_id))
                    conn.commit()
                print(f"[Node 4: CryptoLedger] Offline-first Mode: Simulated successful sync for {event_id}")
                continue
            success = False
            print(f"[Node 4: CryptoLedger] Uploading {event_id} to Appwrite cloud function...")
            try:
                headers = {
                    "X-Appwrite-Project": project_id,
                    "X-Appwrite-Key": api_key,
                }
                files = {
                    "video": (os.path.basename(mp4_path), open(mp4_path, "rb"), "video/mp4")
                }
                data = {
                    "bundle": bundle_json
                }
                url = f"{endpoint}/functions/{func_id}/executions"
                response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
                if response.status_code in [200, 201, 202]:
                    success = True
                    print(f"[Node 4: CryptoLedger] Successfully uploaded event {event_id} to Appwrite cloud function.")
                else:
                    print(f"[Node 4: CryptoLedger] Appwrite function upload failed with status code: {response.status_code}")
            except Exception as e:
                print(f"[Node 4: CryptoLedger] Network exception during cloud upload for {event_id}: {str(e)}")
            with self.lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if success:
                    cursor.execute("""
                        UPDATE cloud_queue 
                        SET status = 'SYNCED', synced_at = ? 
                        WHERE event_id = ?
                    """, (time.time(), event_id))
                else:
                    new_retry = retry_count + 1
                    status = "FAILED" if new_retry >= 3 else "PENDING"
                    cursor.execute("""
                        UPDATE cloud_queue 
                        SET status = ?, retry_count = ? 
                        WHERE event_id = ?
                    """, (status, new_retry, event_id))
                conn.commit()
