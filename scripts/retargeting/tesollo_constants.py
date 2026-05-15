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
        ["ring_base", "ring_mp"],
    ),
]


# Stage 2c (retarget_sequence_feasible): the hand has *two* qualitatively
# different relationships with the cube. Fingertips are *supposed* to touch it
# (that's the whole task). Every other finger part and the palm should *never*
# touch it. We enforce these with two separate CollisionAvoidanceLimit blocks,
# each with its own minimum distance:
#
#   FEASIBILITY_NONTIP_PAIRS  — hard 5 mm clearance for non-tip finger links
#                                and the palm against the cube.
#   FEASIBILITY_TIP_PAIRS     — soft ≈0 mm clearance so tips can rest on the
#                                cube surface during scheduled contact, but the
#                                IK never drives them deep into the box.
_FINGER_TIP_GEOMS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]
_FINGER_NONTIP_GEOMS = [
    f"{finger}_{seg}"
    for finger in ("thumb", "index", "middle", "ring", "little")
    for seg in ("base", "pp", "mp", "dp")
] + ["little_palmar"]
_PALM_GEOMS = ["palm_upper", "palm_lower"]

FEASIBILITY_NONTIP_PAIRS = [
    (_FINGER_NONTIP_GEOMS + _PALM_GEOMS, ["tracked_cube_geom"]),
]
FEASIBILITY_TIP_PAIRS = [
    (_FINGER_TIP_GEOMS, ["tracked_cube_geom"]),
]
# Combined pair list, retained for callers that want a single CollisionAvoidance
# limit with one uniform min-distance (mostly diagnostics / tests).
FEASIBILITY_COLLISION_PAIRS = FEASIBILITY_NONTIP_PAIRS + FEASIBILITY_TIP_PAIRS

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