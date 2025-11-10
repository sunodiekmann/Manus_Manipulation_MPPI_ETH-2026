from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import zmq
import numpy as np
import threading 

import time

from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "output_mocap.xml"

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

        theta = np.pi  # 90 degrees
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

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


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    mocap = ManusSkeletonReceiver()

    configuration = mink.Configuration(model)

    posture_task = mink.PostureTask(model, cost=1e-2)

    fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
    finger_tasks = []
    for finger in fingers:
        task = mink.FrameTask(
            frame_name=finger,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        finger_tasks.append(task)

    collision_pairs = [
        (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["palm_lower", "palm_upper"]),
        (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]),
        (["thumb_base", "index_base", "middle_base", "ring_base", "little_base"], ["thumb_base", "index_base", "middle_base", "ring_base", "little_base"]),
        (["little_tip"], ["ring_base", "ring_before_tip"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        # mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    tasks = [
        posture_task,
        *finger_tasks,
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    scale = 1.25
    common_x_offset = 0.04
    common_y_offset = -0.00
    offsets = {"thumb_tip": np.array([common_x_offset, common_y_offset, 0]),
               "index_tip": np.array([common_x_offset, common_y_offset, 0]), 
               "middle_tip": np.array([common_x_offset, common_y_offset, 0]), 
               "ring_tip": np.array([common_x_offset, common_y_offset, 0]), 
               "little_tip": np.array([common_x_offset, common_y_offset, 0])}

    # define rotations: 90° around Y, then 90° around Z
    r_x = R.from_euler('x', 90, degrees=True)
    r_z = R.from_euler('z', -90, degrees=True)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # configuration.update_from_keyframe("grasp hard")

        # Initialize mocap bodies at their respective sites.
        posture_task.set_target_from_configuration(configuration)
        for finger in fingers:
            mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "geom")

        rate = RateLimiter(frequency=500.0, warn=False)
        dt = rate.dt
        t = 0
        while viewer.is_running():
            # get task target from mocap
            fingertip_positions = mocap.get_fingertip_pos()
            fingertip_quats = mocap.get_fingertip_quat()

            if fingertip_positions is not None and fingertip_quats is not None:
                # TODO: set mocap objects to fingetips from manus
                for i, finger in enumerate(fingers):
                    mocap_id = model.body(f"{finger}_target").mocapid[0]

                    pos = fingertip_positions[i]
                    quat = fingertip_quats[i]

                    data.mocap_pos[mocap_id] = pos.copy() * scale + offsets[finger]
                    # data.mocap_quat[mocap_id] = R.from_quat(quat).as_quat(scalar_first=True)
                    # compose: first rotate around Y, then Z
                    q_rotated = R.from_quat(quat) * r_x * r_z 

                    if finger == "thumb_tip":
                        q_rotated = q_rotated * r_x * r_x

                    # convert back to quaternion (with scalar first)
                    data.mocap_quat[mocap_id] = q_rotated.as_quat(scalar_first=True)

            # Update task target.
            for finger, task in zip(fingers, finger_tasks):
                task.set_target(
                    mink.SE3.from_mocap_name(model, data, f"{finger}_target")
                )

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-5, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt
