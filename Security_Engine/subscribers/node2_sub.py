import zmq
import json

def listen_node2(pipeline_state, stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print("[Node 4: Subscribers] Node 2 Triage subscriber connected to tcp://127.0.0.1:5556")
    msg_count = 0
    while not stop_event.is_set():
        socks = dict(poller.poll(100))
        if socket in socks:
            try:
                msg = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg)
                pipeline_state.update_triage(data)
                msg_count += 1
                crash_prob = data.get("kinematics", {}).get("crash_prob", 0.0)
                if crash_prob > 0.85:
                    print(f"[Node 4: Subscribers] Node 2 alert! High crash_prob: {crash_prob:.2f}")
                elif msg_count % 50 == 0:
                    print(f"[Node 4: Subscribers] Node 2 msg received. Current crash_prob: {crash_prob:.2f}")
            except Exception as e:
                print(f"[Node 4: Subscribers] Exception in Node 2 subscriber: {str(e)}")
    socket.close()
    context.term()
