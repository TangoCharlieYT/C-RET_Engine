import collections
import threading
import cv2

class RingBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = collections.deque(maxlen=300)
        self.is_locked = False
        self.overflow_buffer = []

    def push(self, frame):
        ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        jpeg_bytes = encoded.tobytes()
        with self.lock:
            if not self.is_locked:
                self.buffer.append(jpeg_bytes)
                if len(self.buffer) % 30 == 0:
                    print(f"[Node 4: RingBuffer] Buffer occupancy: {len(self.buffer)}/300 frames")
            else:
                self.overflow_buffer.append(jpeg_bytes)
        return jpeg_bytes

    def snapshot(self):
        with self.lock:
            return list(self.buffer)

    def lock_buffer(self):
        with self.lock:
            self.is_locked = True
            self.overflow_buffer = []
            print("[Node 4: RingBuffer] Buffer is LOCKED. Storing frames in overflow.")

    def unlock_buffer(self):
        with self.lock:
            self.is_locked = False
            self.overflow_buffer = []
            print("[Node 4: RingBuffer] Buffer is UNLOCKED. Resumed normal operations.")

    def get_overflow(self):
        with self.lock:
            return list(self.overflow_buffer)
