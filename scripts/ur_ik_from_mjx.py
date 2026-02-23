from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np
import time

import threading 
import zmq

from scipy.spatial.transform import Rotation as R

import os, sys
# Add repo root to sys.path
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_root)

# from src.robot_ipc_control.controller import 
from src.robot_ipc_control.planning_utils.robot_interface import (
    Robot
)

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "universal_robots_ur5e/scene.xml"


class WristPoseReceiver:
    '''
    Applies to any ZMQ-broadcasted mocap data with fixed shape (21,3) and dtype float32.
    Runs a background thread to continuously receive and update latest data.
    '''
    def __init__(self, port=8001):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(
            zmq.CONFLATE, 1
        )  # Keep only the latest message
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        self.socket.connect(f"tcp://localhost:{port}")

        self._latest_pos = np.zeros(7)

        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                message = self.socket.recv_json()
                data = message["poses"]

                print(data)

                with self._lock:
                    self._latest_pos = np.array(data)
            except zmq.Again:
                time.sleep(0.001)

    def get(self):
        with self._lock:
            return list(self._latest_pos)
       
    def close(self):
        self._running = False
        self._thread.join()
        self.socket.close()


WITH_ROBOT = True

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    KeyboardThread()

    # raise ValueError
    pose_estimator_config = 8001
    mocap = WristPoseReceiver(pose_estimator_config)

    if WITH_ROBOT:
      zmq_ip = "127.0.0.1"
    #   robot_1_config_path = "/home/valentin/git/postdoc/robotic-venv/robot_ipc_control/controller/build/single_arm.json"
      robot_1_config_path = "/home/duplo/git/robohand/src/robot_ipc_control/controller/pc_ur.json"
    #   robot = Robot(robot_1_config_path, zmq_ip, use_gripper=False)
      robot = Robot(robot_1_config_path, zmq_ip)

    # home_pose = np.array([np.pi, -1.5708, 1.5708, 0, 1.5708, 0])
    home_pose = np.array([0, -1.5708, 1.5708, 0, 0, 3.63])

    configuration = mink.Configuration(model)
    configuration.update(home_pose)

    posture_cost = np.ones(model.nv) * 1e-2
    # posture_cost[-2] = 1
    # posture_cost[-1] = 1
    posture_task = mink.PostureTask(model, cost=posture_cost)
    
    damping_cost = np.ones(model.nv) * 0.5
    # damping_cost[-2] = 10
    damping_task = mink.DampingTask(model, damping_cost)

    ee_task = mink.FrameTask(
        # frame_name="palm_site",
        frame_name="ee_attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    max_velocities = {
        "shoulder_pan_joint": np.pi,
        "shoulder_lift_joint": np.pi,
        "elbow_joint": np.pi,
        "wrist_1_joint": np.pi,
        "wrist_2_joint": np.pi,
        "wrist_3_joint": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)

    collision_pairs = [
        # (["wrist_3_link", "hand_dummy"], ["floor", "wall_side_right", "wall_back", "wall_front"]),
    ]

    limits = [
        # velocity_limit,
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    tasks = [
        # damping_task,
        posture_task,
        ee_task,
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    pos_offset = np.array([-0.4, -0.3, 0.3])
    orientation_offset = None

    latest_joint_data = home_pose

    # define rotations: 90° around Y, then 90° around Z
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize mocap bodies at their respective sites.
        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "target", "ee_attachment_site", "site")

        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.dt
        t = 0
        while viewer.is_running():
            # get task target from mocap
            wrist_pose = mocap.get()

            if wrist_pose is not None and np.linalg.norm(wrist_pose)> 1e-3:
                wrist_position = wrist_pose[:3] + pos_offset
                wrist_quat = wrist_pose[3:].copy()
                
                mocap_id = model.body("target").mocapid[0]

                data.mocap_pos[mocap_id] = wrist_position.copy()

                # R_mocap_t = R.from_quat(wrist_quat, scalar_first=True) # Assuming (w, x, y, z)
                # R_target = R_mocap_t * orientation_offset
                # wrist_quat_target = R_target.as_quat(scalar_first=True) 
                # data.mocap_quat[mocap_id] = wrist_quat_target.copy()
                # rotate by 90 deg around world Z
                R_mocap = R.from_quat(wrist_quat, scalar_first=True)  # assume (w, x, y, z)
                R_z = R.from_euler("z", -90, degrees=True)
                R_target = R_z * R_mocap
                # R_target = R_mocap
                # rotate 90 deg around the target's own local Z axis (apply on right)
                R_local = R.from_euler("z", -160, degrees=True)
                R_target_local = R_target * R_local
                # R_target_local = R_target 
                wrist_quat_target = R_target_local.as_quat(scalar_first=True)
                data.mocap_quat[mocap_id] = wrist_quat_target.copy()

            # Update task target.
            ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

            posture_task.set_target(latest_joint_data)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            if WITH_ROBOT:
                latest_joint_data = data.qpos.copy()
                command = {"target_q": list(latest_joint_data)}
                print(f"Sending target pose: {command}")
                robot.controller_publisher.send_json(command)

            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt

            time.sleep(0.01)
