import zmq
import json

def listen_node3(pipeline_state, stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5557")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    print("[Node 4: Subscribers] Node 3 Risk subscriber connected to tcp://127.0.0.1:5557")
    msg_count = 0
    while not stop_event.is_set():
        socks = dict(poller.poll(100))
        if socket in socks:
            try:
                msg = socket.recv_string(flags=zmq.NOBLOCK)
                data = json.loads(msg)
                pipeline_state.update_risk(data)
                msg_count += 1
                risk_score = data.get("risk_score", 0.0)
                if risk_score > 0.85:
                    print(f"[Node 4: Subscribers] Node 3 alert! High risk_score: {risk_score:.2f}")
                elif msg_count % 50 == 0:
                    print(f"[Node 4: Subscribers] Node 3 msg received. Current risk_score: {risk_score:.2f}")
            except Exception as e:
                print(f"[Node 4: Subscribers] Exception in Node 3 subscriber: {str(e)}")
    socket.close()
    context.term()
