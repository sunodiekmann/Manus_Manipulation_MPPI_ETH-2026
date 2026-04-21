import threading 
import zmq
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

class ManusSkeletonReceiver:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (25,7).
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(
        self,
        port=8000,
        left_glove_serial_number="5186fac7",
        right_glove_serial_number="dc8b6ba8"
    ):
        self.left_serial_number = left_glove_serial_number
        self.right_serial_number = right_glove_serial_number

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(
            zmq.CONFLATE, 1
        )  # Keep only the latest message
        self.socket.connect(f"tcp://localhost:{port}")

        self._latest_pos = {
            self.left_serial_number: None,
            self.right_serial_number: None,
        }
        self._latest_quat = {
            self.left_serial_number: None,
            self.right_serial_number: None,
        }
        self._latest_timestamp = {
            self.left_serial_number: None,
            self.right_serial_number: None,
        }

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.fingertip_indices = [4, 9, 14, 19, 24]

    def _process_packet(self, data):
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

        serial_number = data[0]
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

        return serial_number, pos, quats

    def _recv_loop(self):
        while self._running:
            try:
                message = self.socket.recv()
                recv_timestamp = time.time()
                message = message.decode('utf-8')
                data = message.split(",")

                if len(data) == 176:
                    packets = [data]
                elif len(data) == 352:
                    packets = [data[0:176], data[176:352]]
                else:
                    continue

                for packet in packets:
                    serial_number = packet[0]

                    if serial_number not in [self.left_serial_number, self.right_serial_number]:
                        continue

                    serial_number, pos, quats = self._process_packet(packet)

                    with self._lock:
                        self._latest_pos[serial_number] = pos
                        self._latest_quat[serial_number] = quats
                        self._latest_timestamp[serial_number] = recv_timestamp

            except zmq.Again:
                time.sleep(0.001)

    def get(self, glove="right"):
        serial_number = self.right_serial_number if glove == "right" else self.left_serial_number

        with self._lock:
            if self._latest_pos[serial_number] is not None:
                return {"result": self._latest_pos[serial_number].copy(), 
                        "timestamp": self._latest_timestamp[serial_number],
                        "status": "recording"}
            else:
                return {"result": None, 
                        "timestamp": None,
                        "status": "no data"}
            
    def get_fingertip_pos(self, glove="right"):
        serial_number = self.right_serial_number if glove == "right" else self.left_serial_number

        with self._lock:
            if self._latest_pos[serial_number] is not None:
                return self._latest_pos[serial_number][self.fingertip_indices, :]
            else:
                return None
            
    def get_fingertip_quat(self, glove="right"):
        serial_number = self.right_serial_number if glove == "right" else self.left_serial_number

        with self._lock:
            if self._latest_quat[serial_number] is not None:
                return self._latest_quat[serial_number][self.fingertip_indices, :]
            else:
                return None

    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()