import os
import sys
import threading
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipeline_state import PipelineState
from core.ring_buffer import RingBuffer
from core.crypto_ledger import CryptoLedger
from core.evidence_locker import EvidenceLocker
from core.ueba_engine import UEBAEngine
from subscribers.node1_sub import listen_node1
from subscribers.node2_sub import listen_node2
from subscribers.node3_sub import listen_node3

def main():
    print("[Node 4] Initializing Security & Locker Engine...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, 'config.env')
    load_dotenv(env_path)
    secret_key = os.getenv("HMAC_SECRET", "cret_edge_mesh_key_2025")
    appwrite_config = {
        "APPWRITE_ENDPOINT": os.getenv("APPWRITE_ENDPOINT", ""),
        "APPWRITE_PROJECT_ID": os.getenv("APPWRITE_PROJECT_ID", ""),
        "APPWRITE_API_KEY": os.getenv("APPWRITE_API_KEY", ""),
        "APPWRITE_FUNCTION_ID": os.getenv("APPWRITE_FUNCTION_ID", "")
    }
    db_path = os.path.join(base_dir, "metadata.db")
    output_dir = os.path.abspath(os.path.join(base_dir, "..", "assets", "evidence_locker"))
    csv_path = os.path.abspath(os.path.join(base_dir, "..", "assets", "can_telemetry.csv"))
    print(f"[Node 4] Base directory: {base_dir}")
    print(f"[Node 4] Config env loaded from: {env_path}")
    print(f"[Node 4] SQLite Queue database: {db_path}")
    print(f"[Node 4] Evidence locker path: {output_dir}")
    print(f"[Node 4] CAN telemetry log: {csv_path}")
    pipeline_state = PipelineState()
    ring_buffer = RingBuffer()
    crypto_ledger = CryptoLedger(
        secret_key=secret_key,
        db_path=db_path,
        appwrite_config=appwrite_config
    )
    evidence_locker = EvidenceLocker(
        ring_buffer=ring_buffer,
        pipeline_state=pipeline_state,
        crypto_ledger=crypto_ledger,
        output_dir=output_dir
    )
    ueba_engine = UEBAEngine(csv_path=csv_path, pipeline_state=pipeline_state)
    stop_event = threading.Event()
    t_node2 = threading.Thread(
        target=listen_node2,
        args=(pipeline_state, stop_event),
        daemon=True
    )
    t_node3 = threading.Thread(
        target=listen_node3,
        args=(pipeline_state, stop_event),
        daemon=True
    )
    t_ueba = threading.Thread(
        target=ueba_engine.run_engine,
        args=(stop_event,),
        daemon=True
    )
    t_sync = threading.Thread(
        target=crypto_ledger.run_sync_worker,
        args=(stop_event,),
        daemon=True
    )
    print("[Node 4] Starting subscriber and simulation threads...")
    t_node2.start()
    t_node3.start()
    t_ueba.start()
    t_sync.start()
    print("[Node 4] Initialization finished. Starting main ingestion block.")
    try:
        listen_node1(pipeline_state, ring_buffer, evidence_locker, stop_event)
    except KeyboardInterrupt:
        print("[Node 4] KeyboardInterrupt detected. Initiating shutdown sequence...")
    finally:
        stop_event.set()
        print("[Node 4] Security Engine shutdown completed.")

if __name__ == "__main__":
    main()
