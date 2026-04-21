import threading
import time
import base64
import cv2
import zmq

from typing import Dict, Optional
import numpy as np
from numpy.typing import NDArray

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.streams.board_pose_filter import BoardPoseFilter


class BoardPoseListener:
    box_filters: Dict[str, BoardPoseFilter]

    def __init__(self, box_pose_socket_address: str):
        """
        Initialize the board pose estimator.

        Args:
            box_pose_socket_address: The ZMQ socket address for receiving pose data
        """
        self.box_pose_socket_address = box_pose_socket_address

        self.context = None
        self.box_pose_socket = None

        self.box_filters = {}
        self.frames = {}

        self.pose_timestamps = {}
        self.frame_timestamps = {}

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the board pose estimation thread"""
        try:
            self.context = zmq.Context()
            self.box_pose_socket = self.context.socket(zmq.SUB)
            self.box_pose_socket.setsockopt(zmq.CONFLATE, 1)
            self.box_pose_socket.connect(self.box_pose_socket_address)
            self.box_pose_socket.setsockopt(zmq.SUBSCRIBE, b"")

            self._running = True

            self._thread = threading.Thread(target=self._update_loop)
            self._thread.daemon = True
            self._thread.start()

            return True
        except Exception as e:
            print(f"Failed to start board pose estimator: {e}")
            if hasattr(self, "box_pose_socket") and self.box_pose_socket is not None:
                self.box_pose_socket.close()
            if hasattr(self, "context") and self.context is not None:
                self.context.term()
            return False

    def stop(self):
        """Stop the board pose estimation thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if hasattr(self, "box_pose_socket") and self.box_pose_socket is not None:
            self.box_pose_socket.close()
        if hasattr(self, "context") and self.context is not None:
            self.context.term()

    def get_pose(self, board_id: str) -> Optional[NDArray]:
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].get_pose()
        return None

    def get_timestamp(self, board_id: str):
        with self._lock:
            return self.pose_timestamps.get(board_id, None)

    def get_pose_with_timestamp(self, board_id: str):
        with self._lock:
            if board_id in self.box_filters:
                return {
                    "pose": self.box_filters[board_id].get_pose(),
                    "timestamp": self.pose_timestamps.get(board_id, None),
                }
        return {
            "pose": None,
            "timestamp": None,
        }

    def get_frame(self, camera_id: str) -> Optional[NDArray]:
        with self._lock:
            if camera_id in self.frames:
                return self.frames[camera_id]
        return None

    def get_frame_timestamp(self, camera_id: str):
        with self._lock:
            return self.frame_timestamps.get(camera_id, None)

    def is_stable(self, board_id: str) -> Optional[bool]:
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].is_stable()
        return None

    def get_confidence(self, board_id: str) -> float:
        with self._lock:
            if board_id in self.box_filters:
                return self.box_filters[board_id].get_confidence()
        return None

    def get_tracked_board_ids(self):
        with self._lock:
            return list(self.box_filters.keys())

    def _update_loop(self):
        try:
            while self._running:
                try:
                    data = self.box_pose_socket.recv_json()
                    recv_timestamp = time.time()

                    box_pose_measurement = data.get("poses", None)

                    with self._lock:
                        if box_pose_measurement is not None:
                            for box, measurement in box_pose_measurement.items():
                                box = str(box)
                                if box not in self.box_filters:
                                    self.box_filters[box] = BoardPoseFilter(measurement)

                                self.box_filters[box].update(measurement)
                                self.pose_timestamps[box] = recv_timestamp

                    frame_dict = data.get("images", None)
                    if frame_dict is not None:
                        for camera_id, frame_b64 in frame_dict.items():
                            frame_data = base64.b64decode(frame_b64)
                            np_arr = np.frombuffer(frame_data, np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            with self._lock:
                                self.frames[camera_id] = frame
                                self.frame_timestamps[camera_id] = recv_timestamp

                except zmq.ZMQError as e:
                    if not self._running:
                        break
                    print(f"ZMQ error: {e}")

        except Exception as e:
            print(f"Error in board pose estimator thread: {e}")

        finally:
            if self.box_pose_socket is not None:
                self.box_pose_socket.close()