import threading 
import zmq
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

def right_ergonomics_to_mujoco(ergonomics_data):
    ergonomics_data[1] = ergonomics_data[1] - 90.
    ergonomics_data[2] = ergonomics_data[2]

    for i in [1, 2, 3]:
        ergonomics_data[i * 4] = -ergonomics_data[i*4] + 0.

    tmp = ergonomics_data[0]
    ergonomics_data[0] = ergonomics_data[1]
    ergonomics_data[1] = tmp

    ergonomics_data[0] = -ergonomics_data[0]
    ergonomics_data[1] = -ergonomics_data[1] - 45

    tmp = ergonomics_data[16]
    ergonomics_data[16] = ergonomics_data[17] * 0.5
    ergonomics_data[17] = -tmp

    
    # ergonomics_data[16] = 0
    # ergonomics_data[17] = 0
    # ergonomics_data[18] = 0
    # ergonomics_data[19] = 0

    return ergonomics_data


class ManusErgonomicsReceiver():
    def __init__(self, socket_address="tcp://localhost:8001", update_rate = 1/100.):
        self.socket_address = socket_address

        self.context = None
        self.socket = None

        self.update_rate = update_rate
        self.pose = np.zeros(20)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """Start the receiver thread"""
        try:
            print("started")
            # Initialize ZMQ context and socket
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.setsockopt(
                zmq.CONFLATE, 1
            )  # Keep only the latest message
            self.socket.connect(self.socket_address)
            # self.socket.setsockopt(zmq.SUBSCRIBE, b"")

            self._running = True

            # Create thread and start it
            self._thread = threading.Thread(target=self._loop)
            self._thread.daemon = True
            self._thread.start()

            return True
        
        except Exception as e:
            print(f"Failed to start: {e}")
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
            return False

    def stop(self):
        """Stop the board pose estimation thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        # Clean up ZMQ resources
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()

    def _loop(self):
        try:
            while self._running:
                # Get box pose estimation
                try:
                    #wait for message
                    message = self.socket.recv()
                    #receive the message from the socket
                    message = message.decode('utf-8')
                    # print("Received reply %s" % (message))
                    data = message.split(",") 
                    if len(data) == 40:
                        right_hand_ergonomics = list(map(float,data[20:40]))
                        right_hand_ergonomics = right_ergonomics_to_mujoco(right_hand_ergonomics)

                        with self._lock:
                            self.pose = np.array(right_hand_ergonomics) / 360 * 2 * np.pi

                except zmq.Again:
                    pass

                # Sleep for a bit to avoid hogging CPU
                time.sleep(self.update_rate)
        except Exception as e:
            print(f"Error in board pose estimator thread: {e}")

        finally:
            # Clean up ZMQ socket
            self.socket.close()

    def get_angles_in_rad(self):
         with self._lock:
            return self.pose.tolist()