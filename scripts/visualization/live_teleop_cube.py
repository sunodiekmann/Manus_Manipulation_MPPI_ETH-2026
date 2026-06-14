"""Live teleop using the cube-relative pipeline.

Fingertip targets are placed in cube frame and transformed to world via the
tracked cube pose. Uses mink's gradient IK from `tesollo_mink_ik` (the
offline retargeter switched to SPIDER-style equality IK in the rewrite;
runtime teleop still uses mink for its latency/safety profile).

Usage:
    python -m scripts.visualization.live_teleop_cube
    python -m scripts.visualization.live_teleop_cube --offset 0.0043 0.0064 -0.0268
"""
import sys

import mujoco
import mujoco.viewer
import numpy as np
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

from scripts.retargeting.tesollo_calibration import (
    DEFAULT_WRIST_OFFSET, apply_wrist_offset,
)
from scripts.retargeting.tesollo_constants import TARGET_SCALE
from scripts.retargeting.tesollo_ik_runtime import (
    initialize_targets_from_configuration,
)
from scripts.retargeting.tesollo_mink_ik import (
    MODEL_PATH, _default_quats, ik_frame,
)
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_hand_base_pose,
)
from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle

RIGHT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json"


def _parse_offset(argv):
    if "--offset" in argv:
        i = argv.index("--offset")
        return np.array([float(argv[i+1]), float(argv[i+2]), float(argv[i+3])], dtype=float)
    return DEFAULT_WRIST_OFFSET.copy()


def _fingertips_wrist_to_cube(fingertips_wrist, T_world_wrist, T_world_cube):
    """Convert per-finger positions from wrist frame to cube frame."""
    if fingertips_wrist is None or T_world_wrist is None or T_world_cube is None:
        return None
    T_cube_world = np.linalg.inv(T_world_cube)
    R_wc = T_cube_world[:3, :3]
    t_wc = T_cube_world[:3, 3]
    R_wr = T_world_wrist[:3, :3]
    t_wr = T_world_wrist[:3, 3]
    out = np.zeros((len(fingertips_wrist), 3))
    for i, p_wrist in enumerate(fingertips_wrist):
        p_world = R_wr @ np.asarray(p_wrist, dtype=float) + t_wr
        out[i] = R_wc @ p_world + t_wc
    return out


def main():
    wrist_offset = _parse_offset(sys.argv[1:])
    print(f"Wrist offset: {wrist_offset.tolist()}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

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

    default_quats = _default_quats(len(task_bundle.fingers))

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        initialize_targets_from_configuration(task_bundle)
        initialize_fingertip_target_mocaps(model, data, task_bundle.fingers)

        rate = RateLimiter(frequency=30.0, warn=False)

        try:
            while viewer.is_running():
                live_data = live.get_latest(max_age=0.2)

                fingertip_positions = live_data["right"]["fingertip_positions"]
                ergonomics = live_data["right"]["ergonomics_angles"]
                right_board_pose = live_data["right_board_pose"]
                cube_pose = live_data["cube_pose"]

                T_world_wrist = None
                if right_board_pose is not None:
                    T_world_board = pose_to_matrix(right_board_pose)
                    T_world_wrist = T_world_board @ T_board_from_wrist_right

                T_world_cube = pose_to_matrix(cube_pose) if cube_pose is not None else None

                fingertips_cube = _fingertips_wrist_to_cube(
                    fingertip_positions, T_world_wrist, T_world_cube
                )

                if (fingertips_cube is None or T_world_cube is None
                        or T_world_wrist is None or ergonomics is None):
                    viewer.sync()
                    rate.sleep()
                    continue

                ik_frame(
                    model, data, task_bundle, default_quats,
                    fingertips_cube, T_world_cube, T_world_wrist, ergonomics,
                    scale=TARGET_SCALE, wrist_offset=wrist_offset,
                )

                T_world_robot_base = apply_wrist_offset(T_world_wrist, wrist_offset)
                set_hand_base_pose(model, data, T_world_robot_base)

                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()

        finally:
            live.stop()


if __name__ == "__main__":
    main()
