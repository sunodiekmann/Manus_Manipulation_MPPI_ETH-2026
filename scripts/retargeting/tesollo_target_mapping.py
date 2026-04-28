import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.tesollo_constants import (
    TARGET_SCALE,
    DEFAULT_TARGET_OFFSETS,
    R_X,
    R_Z,
)


def rotate_local_fingertip_quaternion(quat_xyzw, finger: str) -> R:
    q_rotated = R.from_quat(quat_xyzw) * R_X * R_Z
    if finger == "thumb_tip":
        q_rotated = q_rotated * R_X * R_X
    return q_rotated


def wrist_to_world_target(pos_wrist, quat_wrist_xyzw, finger: str, T_world_from_wrist):
    R_world_from_wrist = T_world_from_wrist[:3, :3]
    t_world_from_wrist = T_world_from_wrist[:3, 3]

    pos_world = R_world_from_wrist @ np.asarray(pos_wrist, dtype=float) + t_world_from_wrist

    q_local = rotate_local_fingertip_quaternion(quat_wrist_xyzw, finger)
    q_world = R.from_matrix(R_world_from_wrist) * q_local

    return pos_world, q_world


def map_wrist_frame_target(pos_wrist, quat_wrist_xyzw, finger: str, offsets=None, scale: float = TARGET_SCALE):
    if offsets is None:
        offsets = DEFAULT_TARGET_OFFSETS

    pos_target = np.asarray(pos_wrist, dtype=float) * scale + offsets[finger]
    q_target = rotate_local_fingertip_quaternion(quat_wrist_xyzw, finger)
    return pos_target, q_target


def map_cube_frame_target(
    pos_cube,
    quat_wrist_xyzw,
    finger: str,
    T_world_from_cube,
    T_world_from_wrist,
    offsets=None,
    scale: float = TARGET_SCALE,
):
    """Map a fingertip position expressed in cube frame to a world-frame IK target.

    Position is derived from cube-frame coordinates so targets track the cube
    regardless of wrist-tracking noise.  Orientation is derived from the wrist
    frame (same convention as map_world_frame_target) so the IK sees the same
    orientation cost landscape as the baseline and converges to the same local
    minimum.
    """
    if offsets is None:
        offsets = DEFAULT_TARGET_OFFSETS

    pos_cube_adjusted = np.asarray(pos_cube, dtype=float) * scale + offsets[finger]

    R_world_cube = T_world_from_cube[:3, :3]
    t_world_cube = T_world_from_cube[:3, 3]
    pos_world = R_world_cube @ pos_cube_adjusted + t_world_cube

    # Use wrist rotation for orientation — keeps the same target quat as the
    # world-frame baseline so the IK converges to the same local minimum.
    R_world_wrist = T_world_from_wrist[:3, :3]
    q_local = rotate_local_fingertip_quaternion(quat_wrist_xyzw, finger)
    q_world = R.from_matrix(R_world_wrist) * q_local

    return pos_world, q_world


def map_world_frame_target(
    pos_wrist,
    quat_wrist_xyzw,
    finger: str,
    T_world_from_wrist,
    offsets=None,
    scale: float = TARGET_SCALE,
):
    if offsets is None:
        offsets = DEFAULT_TARGET_OFFSETS

    # Apply scale and offset in wrist frame first, then transform to world.
    # This mirrors map_wrist_frame_target — offsets are wrist-relative, not world-relative.
    pos_wrist_adjusted = np.asarray(pos_wrist, dtype=float) * scale + offsets[finger]
    pos_world, q_world = wrist_to_world_target(
        pos_wrist_adjusted,
        quat_wrist_xyzw,
        finger,
        T_world_from_wrist,
    )
    return pos_world, q_world