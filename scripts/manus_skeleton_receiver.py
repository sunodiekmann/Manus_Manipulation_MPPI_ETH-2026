import threading 
import zmq
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

class ManusSkeletonReceiver:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8000, glove_serial_number="dc8b6ba8"):
        self.serial_number = glove_serial_number

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(
            zmq.CONFLATE, 1
        )  # Keep only the latest message
        self.socket.connect(f"tcp://localhost:{port}")

        self._latest_pos = None
        self._latest_quat = None

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.fingertip_indices = [4, 9, 14, 19, 24]

    def _recv_loop(self):

        # ordered as thumb, index, middle, ring, little
        # chain_order = [
        #     0, 21, 22, 23, 24, 
        #     1, 2, 3, 4, 5,
        #     6, 7, 8, 9, 10,
        #     16, 17, 18, 19, 20,
        #     11, 12, 13, 14, 15,
        # ]
        chain_order = [
            0, 1, 2, 3, 4, 
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,
        ]
        
        Ry = R.from_euler('y', -90, degrees=True).as_matrix()

        while self._running:
            try:
                message = self.socket.recv()
                message = message.decode('utf-8')
                data = message.split(",")

                assert len(data) == 176

                serial_number = data[0]
                assert serial_number == self.serial_number
                floats = np.array(data[1:], dtype=float).reshape(-1, 7)

                pos = floats[:, :3]
                quats = floats[:, 3:]

                # --- Remove wrist rotation ---
                R_wrist = R.from_quat(quats[0]).as_matrix()
                R_inv = R_wrist.T  # inverse rotation
                pos = (R_inv @ (pos - pos[0]).T).T  # unrotate & recenter at wrist
                rotmats = R_inv @ R.from_quat(quats).as_matrix()

                # --- Apply additional 90° rotation around world y-axis ---
                pos = (Ry @ pos.T).T
                rotmats = Ry @ rotmats
                quats = R.from_matrix(rotmats).as_quat()

                pos = pos[chain_order, :]
                quats = quats[chain_order, :]

                with self._lock:
                    self._latest_pos = pos
                    self._latest_quat = quats
            except zmq.Again:
                time.sleep(0.001)

    def get(self):
        with self._lock:
            if self._latest_pos is not None:
                return {"result": self._latest_pos.copy(), "status": "recording"}
            else:
                return {"result": None, "status": "no data"}
            
    def get_fingertip_pos(self):
        with self._lock:
            if self._latest_pos is not None:
                return self._latest_pos[self.fingertip_indices, :]
            else:
                return None
            
    def get_fingertip_quat(self):
        with self._lock:
            if self._latest_quat is not None:
                return self._latest_quat[self.fingertip_indices, :]
            else:
                return None


    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()
