from pathlib import Path

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
from scripts.utils.transforms import pose_to_matrix
from scripts.utils.calibration_io import load_calibration_matrix

from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.retargeting.tesollo_constants import TARGET_SCALE, DEFAULT_TARGET_OFFSETS
from scripts.retargeting.tesollo_runtime_baseline import (
    initialize_targets_from_configuration,
    update_fingertip_targets_world_frame,
    update_relative_finger_targets,
    reset_relative_finger_targets,
    update_posture_target_from_full_qpos,
    update_finger_frame_tasks_from_mocaps,
    solve_ik_step,
)
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_hand_base_pose,
    set_mocap_body_pose_from_matrix,
)

RIGHT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json"


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

    T_board_from_wrist_right = load_calibration_matrix(RIGHT_CALIB_JSON)

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

        rate = RateLimiter(frequency=500.0, warn=False)

        try:
            while viewer.is_running():
                live_data = live.get_latest(max_age=0.2)

                fingertip_positions = live_data["right"]["fingertip_positions"]
                fingertip_quats = live_data["right"]["fingertip_quats"]
                angles = live_data["right"]["ergonomics_angles"]

                right_board_pose = live_data["right_board_pose"]
                cube_pose = live_data["cube_pose"]

                T_world_from_wrist = None
                if right_board_pose is not None:
                    T_world_from_board = pose_to_matrix(right_board_pose)
                    T_world_from_wrist = T_world_from_board @ T_board_from_wrist_right

                if cube_pose is not None:
                    T_world_from_cube = pose_to_matrix(cube_pose)
                    set_mocap_body_pose_from_matrix(
                        model,
                        data,
                        body_name="tracked_cube",
                        T_world_from_body=T_world_from_cube,
                    )

                if T_world_from_wrist is not None:
                    set_hand_base_pose(model, data, T_world_from_wrist)

                has_targets = update_fingertip_targets_world_frame(
                    model=model,
                    data=data,
                    task_bundle=task_bundle,
                    fingertip_positions=fingertip_positions,
                    fingertip_quats=fingertip_quats,
                    T_world_from_wrist=T_world_from_wrist,
                    offsets=DEFAULT_TARGET_OFFSETS,
                    scale=TARGET_SCALE,
                )

                if has_targets:
                    update_relative_finger_targets(task_bundle)
                else:
                    reset_relative_finger_targets(task_bundle)

                update_posture_target_from_full_qpos(task_bundle, data, angles)
                update_finger_frame_tasks_from_mocaps(model, data, task_bundle)

                solve_ik_step(task_bundle, dt=rate.dt)

                if T_world_from_wrist is not None:
                    set_hand_base_pose(model, data, T_world_from_wrist)

                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()

        finally:
            live.stop()


if __name__ == "__main__":
    main()