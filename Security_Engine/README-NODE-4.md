# Node 4: The Security & Locker Node

This node handles:
1. High-speed raw video ingestion and compression using a memory-guaranteed ring buffer.
2. Thread-safe async listeners for Node 2 and Node 3 telemetry states.
3. CAN bus anomaly detection (Z-score frequency checks and Isolation Forest model).
4. Evidence locking: Dual-flash warning/critical caching and FFmpeg MP4 compilation.
5. HMAC-SHA256 signing of combined data payloads.
6. Offline-first SQLite queue and background sync to Appwrite.
