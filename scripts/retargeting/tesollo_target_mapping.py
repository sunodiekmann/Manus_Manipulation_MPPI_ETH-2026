import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.tesollo_constants import (
    TARGET_SCALE,
    DEFAULT_TARGET_OFFSETS,
    R_X,
    R_Z,
)
from scripts.utils.config import CUBE_SIZE


# Half-extent of the tracked cube (cube-frame, AABB centred at origin).
_CUBE_HALF = CUBE_SIZE / 2

# Small outward margin: keep targets at least this far from the cube surface
# so the position-controlled finger doesn't drive the tip geom into the cube.
_CUBE_SURFACE_MARGIN = 0.002   # 2 mm


def _clamp_target_outside_cube(p_cube: np.ndarray) -> np.ndarray:
    """If p_cube sits inside the cube (or within the surface margin), project
    it onto the nearest cube face plus a small outward margin. Otherwise
    leave it alone. p_cube is expressed in cube frame (AABB centre = origin).
    """
    abs_p = np.abs(p_cube)
    # Outside the cube already?
    if abs_p.max() >= _CUBE_HALF + _CUBE_SURFACE_MARGIN:
        return p_cube
    # Inside (or within margin): project to the dominant face.
    idx = int(np.argmax(abs_p))
    out = p_cube.copy()
    sign = 1.0 if p_cube[idx] >= 0.0 else -1.0
    out[idx] = sign * (_CUBE_HALF + _CUBE_SURFACE_MARGIN)
    return out


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

    R_world_cube = T_world_from_cube[:3, :3]
    t_world_cube = T_world_from_cube[:3, 3]
    R_world_wrist = T_world_from_wrist[:3, :3]

    # Offset is interpreted in wrist frame — the glove-landmark vs robot-tip
    # bias is a finger/wrist-frame quantity, so a constant correction expressed
    # in the wrist frame stays well-aligned regardless of cube orientation.
    # Rotate offset wrist→cube and add to the cube-frame target.
    offset_cube = R_world_cube.T @ (R_world_wrist @ offsets[finger])
    pos_cube_adjusted = np.asarray(pos_cube, dtype=float) * scale + offset_cube

    # Don't ask IK (or downstream MPPI) to drive a fingertip inside the cube —
    # project any inside-cube target onto the nearest face. Calibration alone
    # cannot guarantee outside-cube targets across sequences whose grip is
    # firmer than the statics holds the offsets were fit to.
    pos_cube_adjusted = _clamp_target_outside_cube(pos_cube_adjusted)

    pos_world = R_world_cube @ pos_cube_adjusted + t_world_cube

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