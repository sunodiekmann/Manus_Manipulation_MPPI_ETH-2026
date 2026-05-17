"""Calibration constants shared by the offline retargeter and runtime teleop.

These values are produced by:
    scripts/calibration/optimize_wrist_offset.py    →  DEFAULT_WRIST_OFFSET
    scripts/calibration/optimize_finger_offsets.py  →  OFFLINE_TARGET_OFFSETS

They live in their own module so both the offline pipeline (which has been
rewritten to a SPIDER-style equality IK) and the runtime mink-based teleop
apps can import them without coupling to either retargeter.
"""

from __future__ import annotations

import numpy as np


# Wrist→robot-base translational offset, expressed in wrist frame (metres).
# Compensates for the offset between the Manus wrist landmark and the robot
# freejoint origin. Override via CLI `--offset tx ty tz` in apps that expose it.
DEFAULT_WRIST_OFFSET = np.array([0.0043, 0.0064, -0.0268], dtype=float)


# Per-finger wrist-frame offsets. Closes the systematic glove-landmark vs
# robot-tip-geom gap. Interpreted in wrist frame (applied via
# `map_cube_frame_target` → rotated wrist→cube each frame), so calibration
# generalises across cube orientations.
OFFLINE_TARGET_OFFSETS = {
    "thumb_tip":  np.array([-0.00229, +0.00614, +0.00316]),
    "index_tip":  np.array([+0.00716, +0.00136, -0.00482]),
    "middle_tip": np.array([-0.00832, +0.00678, -0.00896]),
    "ring_tip":   np.array([-0.01872, +0.00496, -0.00579]),
    "little_tip": np.array([-0.00558, +0.02361, +0.00472]),
}


def apply_wrist_offset(T_world_wrist, offset):
    """Translate a wrist transform by `offset` expressed in the wrist frame.

    Returns None if the input is None (frame skipped upstream).
    """
    if T_world_wrist is None:
        return None
    T = np.asarray(T_world_wrist, dtype=float).copy()
    T[:3, 3] = T[:3, :3] @ np.asarray(offset, dtype=float) + T[:3, 3]
    return T
