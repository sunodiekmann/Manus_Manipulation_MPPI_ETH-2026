import threading
import base64
import cv2
import zmq

from typing import Dict, Optional, List
import numpy as np
from numpy.typing import NDArray


class ImageListener:
    """
    ZMQ SUB listener for image frames published as:
      {"timestamp": <float>, "images": { "<camera_id>": "<base64_jpg>", ... } }

    Stores latest decoded frame per camera_id.
    Uses blocking recv (no update_rate needed).
    """

    def __init__(self, image_socket_address: str, conflate: bool = True):
        self.image_socket_address = image_socket_address

        self.context: Optional[zmq.Context] = None
        self.image_socket: Optional[zmq.Socket] = None

        self.frames: Dict[str, NDArray] = {}
        self.last_timestamps: Dict[str, float] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._conflate = conflate

    def start(self) -> bool:
        """Start the image listener thread."""
        try:
            self.context = zmq.Context()
            self.image_socket = self.context.socket(zmq.SUB)

            if self._conflate:
                # Keep only the newest message (ideal for video)
                self.image_socket.setsockopt(zmq.CONFLATE, 1)

            self.image_socket.connect(self.image_socket_address)
            self.image_socket.setsockopt(zmq.SUBSCRIBE, b"")

            self._running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            return True

        except Exception as e:
            print(f"Failed to start ImageListener: {e}")
            try:
                if self.image_socket is not None:
                    self.image_socket.close()
                if self.context is not None:
                    self.context.term()
            except Exception:
                pass
            return False

    def stop(self) -> None:
        """Stop the listener thread and clean up resources."""
        self._running = False

        # Closing the socket will unblock recv_json() in the thread.
        if self.image_socket is not None:
            try:
                self.image_socket.close()
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=1.0)

        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass

    def get_frame(self, camera_id: str) -> Optional[NDArray]:
        """Get the latest decoded BGR frame for a camera_id."""
        camera_id = str(camera_id)
        with self._lock:
            return self.frames.get(camera_id)

    def get_last_timestamp(self, camera_id: str) -> Optional[float]:
        """Get the timestamp of the latest frame stored for a camera_id."""
        camera_id = str(camera_id)
        with self._lock:
            return self.last_timestamps.get(camera_id)

    def get_tracked_camera_ids(self) -> List[str]:
        """Return a list of camera IDs currently seen."""
        with self._lock:
            return list(self.frames.keys())

    def _update_loop(self) -> None:
        """Background loop receiving and decoding images (blocking)."""
        assert self.image_socket is not None

        try:
            while self._running:
                try:
                    data = self.image_socket.recv_json()  # BLOCKING

                    frame_dict = data.get("images", None)
                    if frame_dict is None:
                        # Ignore non-image messages
                        continue

                    ts = float(data.get("timestamp", 0.0))

                    # Decode frames and store
                    with self._lock:
                        for camera_id, frame_b64 in frame_dict.items():
                            try:
                                frame_data = base64.b64decode(frame_b64)
                                np_arr = np.frombuffer(frame_data, np.uint8)
                                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                if frame is None:
                                    continue
                                cid = str(camera_id)
                                self.frames[cid] = frame
                                self.last_timestamps[cid] = ts
                            except Exception:
                                continue

                except zmq.ZMQError as e:
                    # Expected when stopping/closing socket
                    if not self._running:
                        break
                    print(f"ZMQ error in ImageListener: {e}")

        except Exception as e:
            print(f"Error in ImageListener thread: {e}")

        finally:
            try:
                if self.image_socket is not None:
                    self.image_socket.close()
            except Exception:
                pass