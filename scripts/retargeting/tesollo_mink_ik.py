"""Mink-based runtime IK for cube-relative live teleop.

Used by:
    scripts/visualization/live_teleop_cube.py  — live teleop loop
    scripts/calibration/optimize_*.py          — calibration fitters

The *offline* retargeting pipeline uses SPIDER-style equality IK
(`kinematic_retargeting.py`); this module exists separately because the
runtime loop has different latency/safety requirements and benefits from
mink's gradient IK with posture priors.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.tesollo_calibration import (
    DEFAULT_WRIST_OFFSET,
    OFFLINE_TARGET_OFFSETS,
    apply_wrist_offset,
)
from scripts.retargeting.tesollo_constants import TARGET_SCALE
from scripts.retargeting.tesollo_ik_runtime import (
    reset_relative_finger_targets,
    solve_ik_step,
    update_finger_frame_tasks_from_mocaps,
    update_fingertip_targets_cube_frame,
    update_posture_target_from_joint_angles,
    update_relative_finger_targets,
    update_wrist_task_target,
)
from scripts.retargeting.tesollo_mujoco_utils import set_mocap_body_pose_from_matrix
from scripts.utils.transforms import pose_to_matrix


MODEL_PATH = "/home/suno/robohand2/assets/output_mocap.xml"

# Default mink IK loop parameters.
IK_STEPS = 50
IK_DT = 0.002
PLAYBACK_HZ = 30.0
WARMUP_CALLS = 30

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def _default_quats(n: int) -> np.ndarray:
    return np.tile(_IDENTITY_QUAT_XYZW, (n, 1))


def _pin_wrist_in_configuration(task_bundle, T_world_wrist):
    """Force the freejoint qpos to the given wrist pose."""
    T = np.asarray(T_world_wrist, dtype=float)
    pos = T[:3, 3]
    rot_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    q = task_bundle.configuration.q
    q[0:3] = pos
    q[3] = rot_xyzw[3]
    q[4] = rot_xyzw[0]
    q[5] = rot_xyzw[1]
    q[6] = rot_xyzw[2]


def _matrix_to_pose(T):
    """4×4 matrix → [x,y,z,qw,qx,qy,qz] pose vector."""
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return np.array([pos[0], pos[1], pos[2],
                     quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _set_cube_mocap(model, data, cube_pose):
    if cube_pose is None:
        return
    try:
        T = pose_to_matrix(np.asarray(cube_pose, dtype=float))
        set_mocap_body_pose_from_matrix(model, data, "tracked_cube", T)
    except Exception:
        pass


def ik_frame(
    model, mj_data, task_bundle, default_quats,
    fingertips_cube, T_world_cube, T_world_wrist, ergonomics,
    *, scale=TARGET_SCALE, wrist_offset=DEFAULT_WRIST_OFFSET,
):
    """One frame of cube-relative mink IK.

    `wrist_offset` is a translation in wrist-frame coords applied to the robot
    freejoint pose. Targets remain anchored to the cube; only the robot base
    shifts.

    Returns configuration.q if targets are valid (cube visible + wrist
    tracked), otherwise a NaN-filled array.
    """
    T_world_robot_base = apply_wrist_offset(T_world_wrist, wrist_offset)

    _set_cube_mocap(model, mj_data, None if T_world_cube is None else
                    _matrix_to_pose(T_world_cube))

    has_targets = update_fingertip_targets_cube_frame(
        model=model,
        data=mj_data,
        task_bundle=task_bundle,
        fingertip_positions_cube=fingertips_cube,
        fingertip_quats=default_quats,
        T_world_from_cube=T_world_cube,
        T_world_from_wrist=T_world_robot_base,
        offsets=OFFLINE_TARGET_OFFSETS,
        scale=scale,
    )

    if has_targets:
        update_relative_finger_targets(task_bundle)
    else:
        reset_relative_finger_targets(task_bundle)

    update_posture_target_from_joint_angles(task_bundle, ergonomics, T_world_robot_base)
    update_finger_frame_tasks_from_mocaps(model, mj_data, task_bundle)
    update_wrist_task_target(task_bundle, T_world_robot_base)

    for _ in range(IK_STEPS):
        solve_ik_step(task_bundle, dt=IK_DT)

    mj_data.qpos[: task_bundle.configuration.q.shape[0]] = task_bundle.configuration.q
    mujoco.mj_forward(model, mj_data)

    return task_bundle.configuration.q.copy()
