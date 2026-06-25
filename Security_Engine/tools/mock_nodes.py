import zmq
import json
import time
import random
import threading

def run_mock_node2():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5556")
    print("[Mock Nodes] Node 2 publisher bound to tcp://127.0.0.1:5556")
    start_time = time.time()
    msg_count = 0
    while True:
        elapsed = time.time() - start_time
        cycle = elapsed % 40.0
        if 15.0 <= cycle <= 20.0:
            crash_prob = random.uniform(0.45, 0.75)
        elif 20.0 < cycle <= 25.0:
            crash_prob = random.uniform(0.88, 0.98)
        else:
            crash_prob = random.uniform(0.01, 0.15)
        data = {
            "keep_frame": True,
            "anomaly_flags": {"low_conf": False, "glare": False},
            "kinematics": {
                "v_rel": random.uniform(5.0, 25.0),
                "crash_prob": round(crash_prob, 2)
            }
        }
        socket.send_string(json.dumps(data))
        msg_count += 1
        if crash_prob > 0.85:
            print(f"[Mock Nodes] Node 2 publishing collision spike: crash_prob={crash_prob:.2f}")
        elif msg_count % 50 == 0:
            print(f"[Mock Nodes] Node 2 publishing baseline: crash_prob={crash_prob:.2f}")
        time.sleep(0.1)

def run_mock_node3():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5557")
    print("[Mock Nodes] Node 3 publisher bound to tcp://127.0.0.1:5557")
    msg_count = 0
    while True:
        data = {
            "context_embedding": [random.uniform(-1, 1) for _ in range(128)],
            "risk_score": round(random.uniform(0.05, 0.25), 2),
            "recommended_action": "nominal"
        }
        socket.send_string(json.dumps(data))
        msg_count += 1
        if msg_count % 50 == 0:
            print(f"[Mock Nodes] Node 3 publishing baseline: risk_score={data['risk_score']:.2f}")
        time.sleep(0.2)

if __name__ == "__main__":
    print("[Mock Nodes] Initializing mock publisher threads...")
    t2 = threading.Thread(target=run_mock_node2, daemon=True)
    t3 = threading.Thread(target=run_mock_node3, daemon=True)
    t2.start()
    t3.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Mock Nodes] Shutting down mock publishers...")
