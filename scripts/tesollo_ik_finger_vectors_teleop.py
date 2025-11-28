from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np

from scipy.spatial.transform import Rotation as R
from manus_skeleton_receiver import ManusSkeletonReceiver
from manus_ergonomics_receiver import ManusErgonomicsReceiver


_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "output_mocap.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    mocap = ManusSkeletonReceiver()
    ergonomics = ManusErgonomicsReceiver()
    ergonomics.start()

    configuration = mink.Configuration(model)

    posture_task = mink.PostureTask(model, cost=5e-2)

    finger_pairs = [
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

    fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
    finger_tasks = []
    for finger in fingers:
        # if finger == "thumb_tip":
        task = mink.FrameTask(
            frame_name=finger,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.01,
            lm_damping=1.0,
        )
        finger_tasks.append(task)

    relative_finger_tasks = []
    for idx1, idx2 in finger_pairs:
        task = mink.RelativeFrameTask(
            frame_name=fingers[idx1],
            frame_type="geom",
            root_name=fingers[idx2],
            root_type="geom",
            position_cost=1,
            orientation_cost=.0,
            lm_damping=0.1,
        )
        relative_finger_tasks.append(task)

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
        *relative_finger_tasks
    ]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    scale = 1.
    common_x_offset = 0.04
    common_y_offset = -0.00
    common_z_offset = 0.05
    offsets = {"thumb_tip": np.array([common_x_offset, common_y_offset, common_z_offset]),
               "index_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
               "middle_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
               "ring_tip": np.array([common_x_offset, common_y_offset, common_z_offset]), 
               "little_tip": np.array([common_x_offset, common_y_offset, common_z_offset])}
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

            angles = ergonomics.get_angles_in_rad()

            if fingertip_positions is not None:
                # set mocap objects to fingetips from manus
                for i, finger in enumerate(fingers):
                    mocap_id = model.body(f"{finger}_target").mocapid[0]

                    pos = fingertip_positions[i]
                    data.mocap_pos[mocap_id] = pos.copy() * scale + offsets[finger]

                    quat = fingertip_quats[i]
                    q_rotated = R.from_quat(quat) * r_x * r_z 

                    if finger == "thumb_tip":
                        q_rotated = q_rotated * r_x * r_x

                    # convert back to quaternion (with scalar first)
                    data.mocap_quat[mocap_id] = q_rotated.as_quat(scalar_first=True)


                for (idx1, idx2), task in zip(finger_pairs, relative_finger_tasks):
                    finger_1 = fingers[idx1]
                    finger_2 = fingers[idx2]

                    T_pm = configuration.get_transform(
                        f"{finger_1}_target", "body", f"{finger_2}_target", "body"
                    )

                    task.set_target(T_pm)
            else:
                for (idx1, idx2), task in zip(finger_pairs, relative_finger_tasks):
                    T = mink.SE3.identity()
                    task.set_target(T)

            if angles is not None:
                posture_task.set_target(angles)

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
