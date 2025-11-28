#!/usr/bin/env python3

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import zmq
import numpy as np
import threading 

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from scipy.spatial.transform import Rotation as R

from manus_skeleton_receiver import ManusSkeletonReceiver
from manus_ergonomics_receiver import ManusErgonomicsReceiver

import time

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keyboard_util import KeyboardThread, get_keyboard_flag, reset_keyboard_flag


_HERE = Path(__file__).parent
_XML = _HERE / ".." / "output_mocap.xml"

class MujocoRetargeting:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())

        self.mocap = ManusSkeletonReceiver()
        self.ergonomics = ManusErgonomicsReceiver()
        self.ergonomics.start()

        self.configuration = mink.Configuration(self.model)

        self.posture_task = mink.PostureTask(self.model, cost=5e-2)

        self.finger_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),

            # (1, 2),
            # (1, 3),
            # (1, 4),

            # (2, 3),
            # (2, 4),

            # (3, 4),
        ]

        self.fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
        self.finger_tasks = []
        for finger in self.fingers:
            # if finger == "thumb_tip":
            task = mink.FrameTask(
                frame_name=finger,
                frame_type="geom",
                position_cost=1.0,
                orientation_cost=0.01,
                lm_damping=1.0,
            )
            self.finger_tasks.append(task)

        self.relative_finger_tasks = []
        for idx1, idx2 in self.finger_pairs:
            task = mink.RelativeFrameTask(
                frame_name=self.fingers[idx1],
                frame_type="geom",
                root_name=self.fingers[idx2],
                root_type="geom",
                position_cost=1,
                orientation_cost=.0,
                lm_damping=0.1,
            )
            self.relative_finger_tasks.append(task)

        collision_pairs = [
            (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["palm_lower", "palm_upper"]),
            (["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"], ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]),
            (["thumb_base", "index_base", "middle_base", "ring_base", "little_base"], ["thumb_base", "index_base", "middle_base", "ring_base", "little_base"]),
            (["little_tip"], ["ring_base", "ring_before_tip"]),
        ]

        self.limits = [
            mink.ConfigurationLimit(model=self.model),
            # mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
        ]

        self.tasks = [
            self.posture_task,
            *self.finger_tasks,
            *self.relative_finger_tasks
        ]

        self.model = self.configuration.model
        self.data = self.configuration.data
        self.solver = "daqp"

        self.scale = 1.
        common_x_offset = 0.04
        common_y_offset = -0.00
        common_z_offset = 0.05
        self.offsets = {"thumb_tip": np.array([common_x_offset, common_y_offset, common_z_offset]),
                "index_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
                "middle_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
                "ring_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
                "little_tip": np.array([common_x_offset, common_y_offset, common_z_offset])}
        self.r_x = R.from_euler('x', 90, degrees=True)
        self.r_z = R.from_euler('z', -90, degrees=True)
        
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        self._latest_joint_data = []

    def run(self):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)

            # configuration.update_from_keyframe("grasp hard")

            # Initialize mocap bodies at their respective sites.
            self.posture_task.set_target_from_configuration(self.configuration)
            for finger in self.fingers:
                mink.move_mocap_to_frame(self.model, self.data, f"{finger}_target", finger, "geom")

            rate = RateLimiter(frequency=500.0, warn=False)
            dt = rate.dt
            t = 0
            while viewer.is_running() and self._running:
                # get task target from mocap
                fingertip_positions = self.mocap.get_fingertip_pos()
                fingertip_quats = self.mocap.get_fingertip_quat()

                angles = self.ergonomics.get_angles_in_rad()

                if fingertip_positions is not None:
                    # set mocap objects to fingetips from manus
                    for i, finger in enumerate(self.fingers):
                        mocap_id = self.model.body(f"{finger}_target").mocapid[0]

                        pos = fingertip_positions[i]
                        self.data.mocap_pos[mocap_id] = pos.copy() * self.scale + self.offsets[finger]

                        quat = fingertip_quats[i]
                        q_rotated = R.from_quat(quat) * self.r_x * self.r_z 

                        if finger == "thumb_tip":
                            q_rotated = q_rotated * self.r_x * self.r_x

                        # convert back to quaternion (with scalar first)
                        self.data.mocap_quat[mocap_id] = q_rotated.as_quat(scalar_first=True)


                    for (idx1, idx2), task in zip(self.finger_pairs, self.relative_finger_tasks):
                        finger_1 = self.fingers[idx1]
                        finger_2 = self.fingers[idx2]

                        T_pm = self.configuration.get_transform(
                            f"{finger_1}_target", "body", f"{finger_2}_target", "body"
                        )

                        task.set_target(T_pm)
                else:
                    for (idx1, idx2), task in zip(self.finger_pairs, self.relative_finger_tasks):
                        T = mink.SE3.identity()
                        task.set_target(T)

                if angles is not None:
                    self.posture_task.set_target(angles)

                # Update task target.
                for finger, task in zip(self.fingers, self.finger_tasks):
                    task.set_target(
                        mink.SE3.from_mocap_name(self.model, self.data, f"{finger}_target")
                    )

                vel = mink.solve_ik(self.configuration, self.tasks, rate.dt, self.solver, 1e-5, limits=self.limits)
                self.configuration.integrate_inplace(vel, rate.dt)
                mujoco.mj_camlight(self.model, self.data)

                with self._lock:
                    self._latest_joint_data = self.data.qpos.copy()

                # Visualize at fixed FPS.
                viewer.sync()
                rate.sleep()
                t += dt

    def get(self):
        with self._lock:
            return self._latest_joint_data

        
    def close(self):
        self._running = False
        self._thread.join()        


class JointTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('joint_trajectory_publisher')
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/dg5f_right_controller/joint_trajectory', 5)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.loop)
        self.joint_names = ["rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
                            "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
                            "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
                            "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
                            "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"]
        

        self.mj_ik = MujocoRetargeting()

    def loop(self):
        # update enter receiver
        if get_keyboard_flag():
            reset_keyboard_flag()

            message = JointTrajectory()
            message.joint_names = self.joint_names

            point = JointTrajectoryPoint()

            configuration = self.mj_ik.get()
            point.positions = configuration
            
            point.time_from_start = Duration(sec=0, nanosec=0)

            message.points.append(point)
            self.get_logger().info("Joint Trajectory publish : {}".format(
                configuration))
            self.publisher_.publish(message)


def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = JointTrajectoryPublisher()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    KeyboardThread()

    print("AAAA")

    main()