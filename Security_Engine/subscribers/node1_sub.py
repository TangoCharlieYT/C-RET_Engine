import zmq
import json
import numpy as np

def listen_node1(pipeline_state, ring_buffer, evidence_locker, stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print("[Node 4: Subscribers] Node 1 FrameData subscriber connected to tcp://127.0.0.1:5555")
    frame_count = 0
    while not stop_event.is_set():
        socks = dict(poller.poll(100))
        if socket in socks:
            try:
                parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) < 2:
                    continue
                frame_data = json.loads(parts[0].decode('utf-8'))
                frame_bytes = parts[1]
                shape = frame_data.get("frame_shape", [720, 1280, 3])
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(shape)
                jpeg_bytes = ring_buffer.push(frame)
                pipeline_state.update_frame(frame_data)
                evidence_locker.handle_frame(jpeg_bytes)
                snap = pipeline_state.get_snapshot()
                crash_prob = snap["triage"]["kinematics"].get("crash_prob", 0.0)
                risk_score = snap["risk"].get("risk_score", 0.0)
                ueba_alert = snap["ueba_alert"]
                anomaly_flags = frame_data.get("anomaly_flags", {})
                camera_blinded = anomaly_flags.get("camera_blinded", False)
                ts = frame_data.get("timestamp", 0.0)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[Node 4: Subscribers] Status: crash_prob={crash_prob:.2f}, risk={risk_score:.2f}, ueba={ueba_alert}, glare={camera_blinded}")
                if crash_prob > 0.85 or risk_score > 0.85 or ueba_alert or camera_blinded:
                    evidence_locker.trigger_critical(ts)
                elif 0.40 <= crash_prob <= 0.84:
                    evidence_locker.trigger_warning(ts)
            except Exception as e:
                print(f"[Node 4: Subscribers] Exception in Node 1 subscriber: {str(e)}")
    socket.close()
    context.term()
