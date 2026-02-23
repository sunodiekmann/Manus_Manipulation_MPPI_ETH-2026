from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np
import time

from scipy.spatial.transform import Rotation as R

import csv
from datetime import datetime

import os, sys
# Add repo root to sys.path
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_root)

# from src.robot_ipc_control.controller import 
from src.robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag

from manus_ergonomics_receiver import ManusErgonomicsReceiver


_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "universal_robots_ur5e/scene.xml"


class WristPoseReceiver:
    def __init__(self, config):
        self.estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        self.estimator.start()

    def get_wrist_pose(self, id=None):
        # Receive pose data
        ids = self.estimator.get_tracked_board_ids()

        for box_id in ids:
            box_pose = self.estimator.get_pose(box_id)

            if id is None or box_id == id:
                return box_pose
            
        return None


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    KeyboardThread()

    # raise ValueError
    pose_estimator_config = {"port": 5557}
    mocap = WristPoseReceiver(pose_estimator_config)

    erg_receiver = ManusErgonomicsReceiver()
    erg_receiver.start()

    # home_pose = np.array([np.pi, -1.5708, 1.5708, 0, 1.5708, 0])
    home_pose = np.array([0, -1.5708, 1.5708, 0, 0, 3.63])

    configuration = mink.Configuration(model)
    configuration.update(home_pose)

    posture_cost = np.ones(model.nv) * 1e-2
    posture_cost[-2] = 1
    posture_cost[-1] = 1
    posture_task = mink.PostureTask(model, cost=posture_cost)
    
    damping_cost = np.ones(model.nv) * 0.5
    # damping_cost[-2] = 10
    damping_task = mink.DampingTask(model, damping_cost)

    ee_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1,
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
        velocity_limit,
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    tasks = [
        damping_task,
        posture_task,
        ee_task,
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    active = False

    pos_offset = None
    orientation_offset = None

    latest_joint_data = home_pose

    # define rotations: 90° around Y, then 90° around Z

    all_poses = []

    start_time = time.time()
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # configuration.update_from_keyframe("grasp hard")

        # Initialize mocap bodies at their respective sites.
        posture_task.set_target_from_configuration(configuration)
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=50.0, warn=False)
        dt = rate.dt
        t = 0
        while viewer.is_running():
            if get_keyboard_flag():
                reset_keyboard_flag()
                active = not active

            # get task target from mocap
            l_wrist_pose = mocap.get_wrist_pose(id = '0')
            r_wrist_pose = mocap.get_wrist_pose(id = '1')

            left_joints = erg_receiver.get_left_angles_in_rad()
            right_joints = erg_receiver.get_right_angles_in_rad()

            print(left_joints)
            print(right_joints)

            if l_wrist_pose is not None:
                print(l_wrist_pose)

            if r_wrist_pose is not None:
                print(r_wrist_pose)

            if r_wrist_pose is not None and l_wrist_pose is not None:
                frame_data = {
                    "timestamp": time.time() - start_time,
                    "left_wrist": l_wrist_pose,
                    "right_wrist": r_wrist_pose,
                    "left_joints": left_joints,
                    "right_joints": right_joints
                }

            all_poses.append(frame_data)
            
            time.sleep(0.01)

    filename = f"wrist_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        
        header = ["timestamp"]

        header += [f"l_wrist_{i}" for i in range(7)]
        header += [f"r_wrist_{i}" for i in range(7)]

        header += [f"l_finger_{i}" for i in range(20)]
        header += [f"r_finger_{i}" for i in range(20)]

        # Header
        writer.writerow(
            header
        )

        for entry in all_poses:
            row = [entry["timestamp"]] + \
                list(entry["left_wrist"]) + \
                list(entry["right_wrist"]) + \
                list(entry["left_joints"]) + \
                list(entry["right_joints"])

            writer.writerow(row)

    print(f"Saved {len(all_poses)} frames to {filename}")