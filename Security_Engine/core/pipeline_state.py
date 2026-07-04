import threading

class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.triage = {
            "keep_frame": True,
            "anomaly_flags": {"low_conf": False, "glare": False},
            "kinematics": {"v_rel": 0.0, "crash_prob": 0.0}
        }
        self.risk = {
            "context_embedding": [],
            "risk_score": 0.0,
            "recommended_action": ""
        }
        self.frame_data = {
            "timestamp": 0.0,
            "sensors": {"speed": 0.0, "lat": 0.0, "lon": 0.0},
            "detections": []
        }
        self.ueba_alert = False
        self.is_locked = False
        self.active_event_ts = 0.0

    def update_triage(self, data):
        with self.lock:
            self.triage.update(data)

    def update_risk(self, data):
        with self.lock:
            self.risk.update(data)

    def update_frame(self, data):
        with self.lock:
            self.frame_data.update(data)

    def set_ueba_alert(self, value):
        with self.lock:
            self.ueba_alert = value

    def set_locked(self, value):
        with self.lock:
            self.is_locked = value

    def set_active_event_ts(self, value):
        with self.lock:
            self.active_event_ts = value

    def get_snapshot(self):
        with self.lock:
            return {
                "triage": dict(self.triage),
                "risk": dict(self.risk),
                "frame_data": dict(self.frame_data),
                "ueba_alert": self.ueba_alert,
                "is_locked": self.is_locked,
                "active_event_ts": self.active_event_ts
            }
