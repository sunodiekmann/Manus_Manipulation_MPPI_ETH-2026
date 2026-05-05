import sys

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

from scripts.streams.live_inputs import LiveInputs
from scripts.utils.config import (
    MANUS_PORT,
    LEFT_GLOVE_SERIAL,
    RIGHT_GLOVE_SERIAL,
    POSE_PORT,
    IMG_PORT,
    LEFT_BOARD_ID,
    RIGHT_BOARD_ID,
    CUBE_ID,
)

from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.retargeting.tesollo_constants import TARGET_SCALE
from scripts.retargeting.retarget_sequence_cube_relative import OFFLINE_TARGET_OFFSETS
from scripts.retargeting.tesollo_runtime_baseline import (
    initialize_targets_from_configuration,
    update_fingertip_targets_wrist_frame,
    update_relative_finger_targets,
    reset_relative_finger_targets,
    update_posture_target_from_joint_angles,
    update_finger_frame_tasks_from_mocaps,
    update_wrist_task_target,
    solve_ik_step,
)
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_hand_base_pose,
    set_mocap_body_pose,
)


DEFAULT_WRIST_OFFSET = np.array([0.0043, 0.0064, -0.0268], dtype=float)


def _parse_offset(argv):
    if "--offset" in argv:
        i = argv.index("--offset")
        return np.array([float(argv[i+1]), float(argv[i+2]), float(argv[i+3])], dtype=float)
    return DEFAULT_WRIST_OFFSET.copy()


def main():
    model = mujoco.MjModel.from_xml_path("/home/suno/robohand2/assets/output_mocap.xml")

    live = LiveInputs(
        manus_port=MANUS_PORT,
        left_glove_serial_number=LEFT_GLOVE_SERIAL,
        right_glove_serial_number=RIGHT_GLOVE_SERIAL,
        pose_port=POSE_PORT,
        img_port=IMG_PORT,
        left_board_id=LEFT_BOARD_ID,
        right_board_id=RIGHT_BOARD_ID,
        cube_id=CUBE_ID,
        enable_images=False,
        debug_hz=1.0,
    )
    live.start()

    wrist_offset = _parse_offset(sys.argv[1:])
    print(f"Wrist offset: {wrist_offset.tolist()}")

    # Fixed robot-base pose in wrist frame: identity rotation, offset translation.
    T_wrist_robot_base = np.eye(4)
    T_wrist_robot_base[:3, 3] = wrist_offset

    task_bundle = build_tesollo_task_bundle(model)

    model = task_bundle.configuration.model
    data = task_bundle.configuration.data

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        initialize_targets_from_configuration(task_bundle)
        initialize_fingertip_target_mocaps(model, data, task_bundle.fingers)
        set_mocap_body_pose(model, data, "tracked_cube", [0, 0, -10], [0, 0, 0, 1])

        rate = RateLimiter(frequency=500.0, warn=False)

        try:
            while viewer.is_running():
                live_data = live.get_latest(max_age=0.2)

                fingertip_positions = live_data["right"]["fingertip_positions"]
                fingertip_quats = live_data["right"]["fingertip_quats"]
                angles = live_data["right"]["ergonomics_angles"]

                has_targets = update_fingertip_targets_wrist_frame(
                    model=model,
                    data=data,
                    task_bundle=task_bundle,
                    fingertip_positions=fingertip_positions,
                    fingertip_quats=fingertip_quats,
                    offsets=OFFLINE_TARGET_OFFSETS,
                    scale=TARGET_SCALE,
                )

                if has_targets:
                    update_relative_finger_targets(task_bundle)
                else:
                    reset_relative_finger_targets(task_bundle)

                update_posture_target_from_joint_angles(task_bundle, angles, T_wrist_robot_base)
                update_finger_frame_tasks_from_mocaps(model, data, task_bundle)
                update_wrist_task_target(task_bundle, T_wrist_robot_base)

                solve_ik_step(task_bundle, dt=rate.dt)
                set_hand_base_pose(model, data, T_wrist_robot_base)

                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()

        finally:
            live.stop()


if __name__ == "__main__":
    main()