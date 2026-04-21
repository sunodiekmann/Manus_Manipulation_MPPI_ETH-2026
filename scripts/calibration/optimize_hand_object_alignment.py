import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

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
    LEFT_CALIB_JSON,
    RIGHT_CALIB_JSON,
    CUBE_SIZE,
)
from scripts.utils.transforms import (
    pose_to_matrix,
    transform_points,
    build_transform,
    matrix_to_dict,
)

# ─── CONFIGURE HERE ──────────────────────────────────────────────────────────
GLOVE = "left"           # "left" or "right"
FINGERTIP_RADIUS = 0.008  # meters — approximate fingertip contact sphere radius
NUM_FRAMES_PER_CAPTURE = 10  # frames averaged into one captured sample
# ─────────────────────────────────────────────────────────────────────────────

# Cube face definitions.
# Convention (verify against your tag placement in the camera stream):
#   "right" → +X face   (x = +half_size)
#   "left"  → -X face   (x = -half_size)
#   "top"   → +Z face   (z = +half_size)
# Each entry is (axis_index, sign).
FACE_AXES = {
    "right": (0, +1),
    "left":  (0, -1),
    "top":   (2, +1),
}

# The 4 reference poses defined for the RIGHT hand.
# For the left hand, "right" and "left" face labels are automatically swapped
# (see `mirror_poses` below) because the thumb is on the opposite side.
_POSES_RIGHT = [
    {
        "name": "Pose 1 — thumb right, all others left",
        "fingers": {
            "thumb":  "right",
            "index":  "left",
            "middle": "left",
            "ring":   "left",
            "little": "left",
        },
    },
    {
        "name": "Pose 2 — thumb right, index top, rest left",
        "fingers": {
            "thumb":  "right",
            "index":  "top",
            "middle": "left",
            "ring":   "left",
            "little": "left",
        },
    },
    {
        "name": "Pose 3 — thumb right, index+middle top, rest left",
        "fingers": {
            "thumb":  "right",
            "index":  "top",
            "middle": "top",
            "ring":   "left",
            "little": "left",
        },
    },
    {
        "name": "Pose 4 — all fingers flat on top face (palm down)",
        "fingers": {
            "thumb":  "top",
            "index":  "top",
            "middle": "top",
            "ring":   "top",
            "little": "top",
        },
    },
]

_FACE_MIRROR = {"right": "left", "left": "right", "top": "top"}


def _mirror_poses(poses: list) -> list:
    """Return a copy of the pose list with left/right face labels swapped."""
    mirrored = []
    for p in poses:
        mirrored.append({
            "name": p["name"].replace("right", "LEFT").replace("left", "right").replace("LEFT", "left"),
            "fingers": {f: _FACE_MIRROR[face] for f, face in p["fingers"].items()},
        })
    return mirrored

# Fingertip order in the (5, 3) array returned by live_data[hand]["fingertip_positions"].
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]


# ─── Geometry helpers ────────────────────────────────────────────────────────

def face_target(face: str, half_size: float, radius: float) -> tuple[int, float]:
    """Return (axis_index, target_value) for a fingertip sphere centre touching face."""
    axis, sign = FACE_AXES[face]
    return axis, sign * (half_size + radius)


def fingertips_in_cube_frame(
    fingertips_wrist: np.ndarray,
    T_world_board: np.ndarray,
    T_world_cube: np.ndarray,
    T_board_wrist: np.ndarray,
) -> np.ndarray:
    """Transform (5, 3) wrist-frame fingertips into the cube frame."""
    T_world_wrist = T_world_board @ T_board_wrist
    T_cube_world = np.linalg.inv(T_world_cube)
    tips_world = transform_points(T_world_wrist, fingertips_wrist)
    return transform_points(T_cube_world, tips_world)


# ─── Optimisation ────────────────────────────────────────────────────────────

def cost_fn(
    params: np.ndarray,
    samples: list,
    T_board_wrist_orig: np.ndarray,
    half_size: float,
    radius: float,
    poses: list,
) -> float:
    """
    Sum of squared distances between each fingertip sphere centre and its
    assigned face target, across all captured samples.

    params = [rx, ry, rz, tx, ty, tz]  (6-DOF correction in wrist frame)
    """
    rx, ry, rz, tx, ty, tz = params
    R_corr = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T_correction = build_transform(R_corr, np.array([tx, ty, tz]))
    T_bw = T_board_wrist_orig @ T_correction

    total = 0.0
    for s in samples:
        tips_cube = fingertips_in_cube_frame(
            s["fingertips_wrist"],
            s["T_world_board"],
            s["T_world_cube"],
            T_bw,
        )
        for finger, face in poses[s["pose_idx"]]["fingers"].items():
            fi = FINGER_ORDER.index(finger)
            axis, target = face_target(face, half_size, radius)
            total += (tips_cube[fi, axis] - target) ** 2

    return total


# ─── Averaging helper (same algorithm as estimate_wrist_to_board) ─────────────

def average_transforms(matrices: list[np.ndarray]) -> np.ndarray:
    if len(matrices) == 1:
        return matrices[0].copy()
    translations = np.stack([M[:3, 3] for M in matrices]).mean(axis=0)
    rot_raw = np.stack([M[:3, :3] for M in matrices]).mean(axis=0)
    U, _, Vt = np.linalg.svd(rot_raw)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R_mean
    T[:3, 3] = translations
    return T


# ─── Capture helpers ─────────────────────────────────────────────────────────

def print_cube_frame_positions(
    fingertips_wrist: np.ndarray,
    T_world_board: np.ndarray,
    T_world_cube: np.ndarray,
    T_board_wrist: np.ndarray,
) -> None:
    tips_cube = fingertips_in_cube_frame(
        fingertips_wrist, T_world_board, T_world_cube, T_board_wrist
    )
    half = CUBE_SIZE / 2
    print("  Fingertips in cube frame  (cube half-size = {:.3f} m)".format(half))
    print("  {:<8}  {:>8}  {:>8}  {:>8}".format("finger", "x", "y", "z"))
    for finger, fi in zip(FINGER_ORDER, range(5)):
        p = tips_cube[fi]
        print("  {:<8}  {:+.4f}  {:+.4f}  {:+.4f}".format(finger, p[0], p[1], p[2]))


def try_capture_frame(
    live: LiveInputs,
    glove_key: str,
    board_key: str,
) -> dict | None:
    """Return a single data snapshot, or None if any stream is unavailable."""
    d = live.get_latest(max_age=0.2)
    tips = d[glove_key]["fingertip_positions"]
    bp   = d[board_key]
    cp   = d["cube_pose"]
    if tips is None or bp is None or cp is None:
        return None
    return {
        "fingertips_wrist": tips.copy(),
        "T_world_board":    pose_to_matrix(bp),
        "T_world_cube":     pose_to_matrix(cp),
    }


def capture_averaged_sample(
    live: LiveInputs,
    glove_key: str,
    board_key: str,
    n_frames: int,
) -> dict | None:
    """Collect n_frames snapshots and return their average, or None on failure."""
    tips_buf, board_buf, cube_buf = [], [], []

    print(f"  Collecting {n_frames} frames ", end="", flush=True)
    for _ in range(n_frames):
        frame = try_capture_frame(live, glove_key, board_key)
        if frame is not None:
            tips_buf.append(frame["fingertips_wrist"])
            board_buf.append(frame["T_world_board"])
            cube_buf.append(frame["T_world_cube"])
            print(".", end="", flush=True)
        time.sleep(0.015)
    print()

    if len(tips_buf) < max(3, n_frames // 2):
        print(f"  Only {len(tips_buf)}/{n_frames} frames valid — skipping.")
        return None

    return {
        "fingertips_wrist": np.mean(tips_buf, axis=0),
        "T_world_board":    average_transforms(board_buf),
        "T_world_cube":     average_transforms(cube_buf),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Hand-specific config ──────────────────────────────────────────────────
    if GLOVE == "right":
        glove_key  = "right"
        board_key  = "right_board_pose"
        calib_path = RIGHT_CALIB_JSON
        poses      = _POSES_RIGHT
    else:
        glove_key  = "left"
        board_key  = "left_board_pose"
        calib_path = LEFT_CALIB_JSON
        poses      = _mirror_poses(_POSES_RIGHT)

    # Always optimise from the base calibration, never stack on a previous result.
    with open(calib_path, "r") as f:
        _calib_json = json.load(f)
    T_board_wrist_orig = np.array(_calib_json["T_board_wrist"]["matrix_4x4"], dtype=float)
    print(f"[calibration] {calib_path}: optimising from 'T_board_wrist' (base)")

    half_size = CUBE_SIZE / 2

    print(f"Optimising {GLOVE}-hand wrist→board calibration")
    print(f"Calibration file : {calib_path}")
    print(f"Cube half-size   : {half_size:.4f} m")
    print(f"Fingertip radius : {FINGERTIP_RADIUS:.4f} m")
    print(f"Face axis mapping: right=+X  left=-X  top=+Z  (verify in camera stream)")

    # ── Start streams ─────────────────────────────────────────────────────────
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
        debug_hz=0.0,
    )
    live.start()

    print("\nWaiting for streams to become available...")
    for _ in range(100):
        d = live.get_latest(max_age=0.5)
        if (
            d[glove_key]["fingertip_positions"] is not None
            and d[board_key] is not None
            and d["cube_pose"] is not None
        ):
            print("All streams ready.\n")
            break
        time.sleep(0.1)

    # ── Capture loop ──────────────────────────────────────────────────────────
    samples: list[dict] = []

    try:
        for pose_idx, pose in enumerate(poses):
            print("=" * 60)
            print(f"Pose {pose_idx + 1}/{len(poses)} — {pose['name']}")
            print("  Finger assignments:")
            for finger, face in pose["fingers"].items():
                print(f"    {finger:<8} → {face} face")
            print()
            print("  Place your hand and press:")
            print("    c + Enter  →  capture a sample")
            print("    Enter      →  done with this pose, move to next")
            print("    q + Enter  →  abort")
            print()

            pose_sample_count = 0

            while True:
                # Show live status without blocking
                d = live.get_latest(max_age=0.2)
                missing = []
                if d[glove_key]["fingertip_positions"] is None:
                    missing.append("glove")
                if d[board_key] is None:
                    missing.append("board")
                if d["cube_pose"] is None:
                    missing.append("cube")
                if missing:
                    print(f"  [waiting: {', '.join(missing)}]")

                cmd = input("  Command [c / Enter / q]: ").strip().lower()

                if cmd == "q":
                    print("Aborted.")
                    live.stop()
                    return

                if cmd == "":
                    if pose_sample_count == 0:
                        print("  Capture at least one sample for this pose first.")
                        continue
                    break

                if cmd == "c":
                    frame = capture_averaged_sample(
                        live, glove_key, board_key, NUM_FRAMES_PER_CAPTURE
                    )
                    if frame is None:
                        print("  Capture failed — check streams and try again.")
                        continue

                    pose_sample_count += 1
                    print(f"  Sample {pose_sample_count} captured (pose {pose_idx + 1}).")
                    print_cube_frame_positions(
                        frame["fingertips_wrist"],
                        frame["T_world_board"],
                        frame["T_world_cube"],
                        T_board_wrist_orig,
                    )
                    print()

                    samples.append({**frame, "pose_idx": pose_idx})

    finally:
        live.stop()

    if len(samples) < len(poses):
        print(f"\nOnly {len(samples)} samples collected across {len(poses)} poses — aborting.")
        return

    # ── Optimise ──────────────────────────────────────────────────────────────
    print(f"\nOptimising over {len(samples)} samples ...")

    x0 = np.zeros(6)
    initial_cost = cost_fn(x0, samples, T_board_wrist_orig, half_size, FINGERTIP_RADIUS, poses)

    result = minimize(
        cost_fn,
        x0,
        args=(samples, T_board_wrist_orig, half_size, FINGERTIP_RADIUS, poses),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
    )

    print(f"  Initial residual : {initial_cost:.6f} m²")
    print(f"  Final residual   : {result.fun:.6f} m²")
    print(f"  Converged        : {result.success}  ({result.message})")

    # ── Build corrected matrix ────────────────────────────────────────────────
    rx, ry, rz, tx, ty, tz = result.x
    R_corr = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    T_correction = build_transform(R_corr, np.array([tx, ty, tz]))
    T_board_wrist_optimized = T_board_wrist_orig @ T_correction

    print("\nT_correction (applied in wrist frame):")
    print(np.array_str(T_correction, precision=6, suppress_small=True))
    print("\nRotation magnitude : {:.3f} deg".format(
        np.degrees(np.linalg.norm([rx, ry, rz]))
    ))
    print("Translation magnitude : {:.4f} m".format(
        np.linalg.norm([tx, ty, tz])
    ))

    # ── Per-pose residual breakdown ───────────────────────────────────────────
    print("\nPer-pose mean fingertip-to-face distance (with optimised calib):")
    for pose_idx, pose in enumerate(poses):
        pose_samples = [s for s in samples if s["pose_idx"] == pose_idx]
        if not pose_samples:
            continue
        dists = []
        for s in pose_samples:
            tips_cube = fingertips_in_cube_frame(
                s["fingertips_wrist"],
                s["T_world_board"],
                s["T_world_cube"],
                T_board_wrist_optimized,
            )
            for finger, face in pose["fingers"].items():
                fi = FINGER_ORDER.index(finger)
                axis, target = face_target(face, half_size, FINGERTIP_RADIUS)
                dists.append(abs(tips_cube[fi, axis] - target))
        print(f"  Pose {pose_idx + 1}: mean={np.mean(dists)*1000:.2f} mm  "
              f"max={np.max(dists)*1000:.2f} mm")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(calib_path, "r") as f:
        calib_data = json.load(f)

    calib_data["T_board_wrist_optimized"] = matrix_to_dict(T_board_wrist_optimized)
    calib_data["T_correction"]            = matrix_to_dict(T_correction)

    with open(calib_path, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\nSaved T_board_wrist_optimized → {calib_path}")
    print("Done.")


if __name__ == "__main__":
    main()
