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
from src.robot_ipc_control.planning_utils.robot_interface import (
    Robot
)

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "universal_robots_ur5e/scene.xml"

WITH_ROBOT = True

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    KeyboardThread()

    if WITH_ROBOT:
      zmq_ip = "127.0.0.1"
      robot_1_config_path = "/home/valentin/git/postdoc/robotic-venv/robot_ipc_control/controller/build/single_arm.json"
      robot = Robot(robot_1_config_path, zmq_ip, use_gripper=False)

    # home_pose = np.array([np.pi, -1.5708, 1.5708, 0, 1.5708, 0])

    # [0.04116866737604141, 
    # -1.5661185721731563, 
    # 1.597208325062887, 
    # 0.03624109804119868, 0.03514508157968521, 3.6302497386932373]
    home_pose = np.array([0, -1.5708, 1.5708, 0, 0, 3.63])

    configuration = mink.Configuration(model)
    configuration.update(home_pose)

    posture_task = mink.PostureTask(model, cost=1e-2)
    
    cost = np.ones(model.nv) * 0.5
    damping_task = mink.DampingTask(model, cost)

    ee_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
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
        (["wrist_3_link", "hand_dummy"], ["floor", "wall_side_right", "wall_back", "wall_front"]),
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

        rate = RateLimiter(frequency=100.0, warn=False)
        dt = rate.dt
        t = 0
        prev_time = time.time()
        
        while viewer.is_running():
            robot.update_robot_state()

            if get_keyboard_flag():
                reset_keyboard_flag()
                active = not active

            # Update task target.
            ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

            posture_task.set_target(latest_joint_data)

            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)

            # print(data.qpos)
            # print(robot.robot_state["Q"])

            if active and WITH_ROBOT:
                latest_joint_data = data.qpos.copy()
                command = {"target_q": list(latest_joint_data)}
                # print(f"Sending target pose: {command}")
                robot.controller_publisher.send_json(command)

                print(f"sending at {t}")
                print(latest_joint_data)

            mujoco.mj_camlight(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            t += dt

            print(1/(time.time() - prev_time))
            prev_time = time.time()
            
            # time.sleep(0.01)
