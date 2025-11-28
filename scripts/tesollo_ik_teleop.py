from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np

from scipy.spatial.transform import Rotation as R

from manus_skeleton_receiver import ManusSkeletonReceiver

_HERE = Path(__file__).parent
_XML = _HERE / "../assets/" / "output_mocap.xml"

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
