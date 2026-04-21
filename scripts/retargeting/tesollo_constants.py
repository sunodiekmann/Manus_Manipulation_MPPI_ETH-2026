from scipy.spatial.transform import Rotation as R
import numpy as np


FINGERS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]

FINGER_PAIRS = [
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

COLLISION_PAIRS = [
    (
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"],
        ["palm_lower", "palm_upper"],
    ),
    (
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"],
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"],
    ),
    (
        ["thumb_base", "index_base", "middle_base", "ring_base", "little_base"],
        ["thumb_base", "index_base", "middle_base", "ring_base", "little_base"],
    ),
    (
        ["little_tip"],
        ["ring_base", "ring_before_tip"],
    ),
]

SOLVER_NAME = "daqp"

TARGET_SCALE = 1.0

COMMON_X_OFFSET = 0.04
COMMON_Y_OFFSET = 0.00
COMMON_Z_OFFSET = 0.05

DEFAULT_TARGET_OFFSETS = {
    finger: np.array([COMMON_X_OFFSET, COMMON_Y_OFFSET, COMMON_Z_OFFSET], dtype=float)
    for finger in FINGERS
}

R_X = R.from_euler("x", 90, degrees=True)
R_Z = R.from_euler("z", -90, degrees=True)