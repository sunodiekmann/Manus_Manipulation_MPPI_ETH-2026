from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np
import time

from scipy.spatial.transform import Rotation as R

import os, sys
# Add repo root to sys.path
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(repo_root)

# from src.robot_ipc_control.controller import 
from src.robot_ipc_control.pose_estimation.board_pose_estimator import BoardPoseEstimator
from src.robot_ipc_control.planning_utils.robot_interface import (
    Robot
)

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "universal_robots_ur5e/scene.xml"


class WristPoseReceiver:
    def __init__(self, config):
        self.estimator = BoardPoseEstimator(f"tcp://localhost:{config['port']}")
        self.estimator.start()

    def get_wrist_pose(self):
        # Receive pose data
        ids = self.estimator.get_tracked_board_ids()

        for box_id in ids:
            box_pose = self.estimator.get_pose(box_id)
            return box_pose


WITH_ROBOT = True

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    KeyboardThread()

    # raise ValueError
    pose_estimator_config = {"port": 5557}
    mocap = WristPoseReceiver(pose_estimator_config)

    if WITH_ROBOT:
      zmq_ip = "127.0.0.1"
      robot_1_config_path = "/home/valentin/git/postdoc/robotic-venv/robot_ipc_control/controller/build/single_arm.json"
      robot = Robot(robot_1_config_path, zmq_ip, use_gripper=False)

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
            wrist_pose = mocap.get_wrist_pose()

            if wrist_pose is not None:
                if not active:
                    print("inactive, resetting offset")
                    current_ee_pos = data.site_xpos[data.site("attachment_site").id]
                    pos_offset = wrist_pose[:3] - current_ee_pos
                    
                    # current_ee_mat = R.from_matrix(np.array(data.site_xmat[data.site("attachment_site").id]).reshape((3,3))).as_matrix()
                    # R_current = R.from_quat(wrist_pose[3:], scalar_first=True).as_matrix()

                    wrist_quat = wrist_pose[3:]
                    R_mocap_t = R.from_quat(wrist_quat, scalar_first=True) # Assuming (w, x, y, z)
                    # -------------------------------------------------------------
                    
                    # Get current EE orientation
                    current_ee_mat = np.array(data.site_xmat[data.site("attachment_site").id]).reshape((3,3))
                    R_EE_current = R.from_matrix(current_ee_mat)
                    R_mocap_inv = R_mocap_t.inv()
                    # orientation_offset = R_EE_current * R_mocap_inv
                    orientation_offset = R_mocap_inv * R_EE_current
                                                
                wrist_position = wrist_pose[:3] - pos_offset

                wrist_quat = wrist_pose[3:].copy()
                
                mocap_id = model.body("target").mocapid[0]

                if active:
                    data.mocap_pos[mocap_id] = wrist_position.copy()

                #     # convert back to quaternion (with scalar first)
                    # data.mocap_quat[mocap_id] = wrist_quat.copy()
                    
                    # data.mocap_quat[mocap_id] = R.from_matrix(R_wrist).as_quat(scalar_first=True)
                    R_mocap_t = R.from_quat(wrist_quat, scalar_first=True) # Assuming (w, x, y, z)
                    R_target = R_mocap_t * orientation_offset
                    wrist_quat_target = R_target.as_quat(scalar_first=True) 
                    data.mocap_quat[mocap_id] = wrist_quat_target.copy()

            # Update task target.
            ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

            posture_task.set_target(latest_joint_data)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            if active and WITH_ROBOT:
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
