"""
Build a synchronised, uniformly-sampled sequence dataset from a raw recording.

Input:  recordings/<session>/  (hands.jsonl, ergonomics.jsonl, poses.jsonl, meta.json)
Output: recordings/<session>/processed_sequence.npz

What it does:
  - Reads the three raw JSONL event streams (hand skeleton, ergonomics, poses).
  - Re-samples everything onto a uniform timeline at TARGET_HZ (default 20 Hz).
    Positions and joint angles use linear interpolation; pose quaternions use
    SLERP so rotations are always valid and move at constant angular velocity.
  - Optionally applies a zero-phase Butterworth low-pass filter (FILTER_HZ) to
    all continuous streams after resampling, removing high-frequency noise from
    the pose estimator and glove without introducing phase lag.
  - Applies the wrist→board calibration matrices (T_board_wrist_optimized if
    present, otherwise T_board_wrist) to express hand joints in world frame.
  - Saves a single .npz with arrays for joints (wrist & world frame), wrist
    transforms, ergonomics angles, and raw pose streams.

Usage:
    python -m scripts.processing.build_sequence_dataset                          # uses hardcoded path
    python -m scripts.processing.build_sequence_dataset recordings/<session>     # override path
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from scripts.utils.transforms import pose_to_matrix, transform_points
from scripts.utils.calibration_io import load_calibration_matrix


RAW_RECORDING_DIR = Path("recordings/statics/session_20260421_150421")
OUTPUT_FILENAME = "processed_sequence.npz"

if len(sys.argv) > 1:
    RAW_RECORDING_DIR = Path(sys.argv[1])

LEFT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/left_wrist2board_calibration.json"
RIGHT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json"

TARGET_HZ = 20.0

# ── filter / outlier settings ─────────────────────────────────────────────────
# Tag-pose outlier rejection is done on the RAW rows BEFORE SLERP resampling.
# This matters: post-resample rejection misses outliers that SLERP already
# smeared across several frames (a 26° raw jump becomes 4–8°/frame bumps that
# slip through any reasonable per-frame threshold).
#
# Thresholds are speed-based so they are independent of the raw frame rate.
# Tune these for your motion: typical manipulation tasks have <60 deg/s and
# <0.15 m/s.  False ArUco detections jump at 100–700 deg/s.
OUTLIER_MAX_POS_SPEED_M_S   = 0.05   # 5 cm/s  — cube movements are mostly rotation; any real translation is slow
OUTLIER_MAX_ROT_SPEED_DEG_S = 80.0   # 80 deg/s — false detections: 130–700 deg/s
# After a false detection the tracker takes a few frames to settle back.
# This many raw frames after each rejected frame are also dropped.
OUTLIER_RECOVERY_FRAMES = 4

# Low-pass filter for tag poses (translation + quaternion) after outlier removal.
# 3 Hz removes residual jitter from the pose estimator without blurring the motion.
# Set to None to disable.
FILTER_HZ_POSES = 3.0

# Gentle low-pass for hand joints and ergonomics angles.  The Manus glove signal
# is already clean; this only removes high-frequency quantisation noise.
# Set to None to disable.
FILTER_HZ_HANDS = 8.0


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_stream(rows):
    grouped = {}
    for row in rows:
        stream = row["stream"]
        grouped.setdefault(stream, []).append(row)
    for stream in grouped:
        grouped[stream] = sorted(grouped[stream], key=lambda x: x["timestamp"])
    return grouped


def build_uniform_timeline(start_t, end_t, hz):
    dt = 1.0 / hz
    n = int(np.floor((end_t - start_t) / dt)) + 1
    return start_t + np.arange(n) * dt


# ── interpolation ─────────────────────────────────────────────────────────────

def interp_vector(rows, key, t):
    """Linear interpolation for a vector field across a sorted list of rows."""
    if len(rows) == 0:
        return None
    if len(rows) == 1:
        return np.array(rows[0][key], dtype=float)

    timestamps = np.array([r["timestamp"] for r in rows], dtype=float)

    if t <= timestamps[0]:
        return np.array(rows[0][key], dtype=float)
    if t >= timestamps[-1]:
        return np.array(rows[-1][key], dtype=float)

    right_idx = int(np.searchsorted(timestamps, t))
    left_idx = right_idx - 1

    alpha = (t - timestamps[left_idx]) / (timestamps[right_idx] - timestamps[left_idx])
    x0 = np.array(rows[left_idx][key], dtype=float)
    x1 = np.array(rows[right_idx][key], dtype=float)
    return (1.0 - alpha) * x0 + alpha * x1


def interp_hand_joints(rows, t):
    x = interp_vector(rows, "joints_wrist", t)
    return None if x is None else np.array(x, dtype=float)


def interp_angles(rows, t):
    x = interp_vector(rows, "angles_rad", t)
    return None if x is None else np.array(x, dtype=float)


def interp_pose_slerp(rows, t):
    """
    Interpolate a 7D pose [x,y,z, qw,qx,qy,qz] using linear interp for
    translation and SLERP for the quaternion.
    """
    if len(rows) == 0:
        return None
    if len(rows) == 1:
        return np.array(rows[0]["pose_world"], dtype=float)

    timestamps = np.array([r["timestamp"] for r in rows], dtype=float)

    if t <= timestamps[0]:
        return np.array(rows[0]["pose_world"], dtype=float)
    if t >= timestamps[-1]:
        return np.array(rows[-1]["pose_world"], dtype=float)

    right_idx = int(np.searchsorted(timestamps, t))
    left_idx = right_idx - 1

    alpha = (t - timestamps[left_idx]) / (timestamps[right_idx] - timestamps[left_idx])

    p0 = np.array(rows[left_idx]["pose_world"],  dtype=float)
    p1 = np.array(rows[right_idx]["pose_world"], dtype=float)

    # translation: linear
    trans = (1.0 - alpha) * p0[:3] + alpha * p1[:3]

    # rotation: SLERP  (stored as [qw, qx, qy, qz] → scipy wants [qx, qy, qz, qw])
    q0_wxyz = p0[3:]
    q1_wxyz = p1[3:]
    q0_xyzw = np.array([q0_wxyz[1], q0_wxyz[2], q0_wxyz[3], q0_wxyz[0]])
    q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])

    # ensure we take the short arc
    if np.dot(q0_xyzw, q1_xyzw) < 0:
        q1_xyzw = -q1_xyzw

    slerp = Slerp([0.0, 1.0], Rotation.from_quat([q0_xyzw, q1_xyzw]))
    q_xyzw = slerp(alpha).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    return np.concatenate([trans, q_wxyz])


# ── filtering ─────────────────────────────────────────────────────────────────

def _butter_lowpass_sos(cutoff_hz: float, sample_hz: float, order: int = 4):
    nyq = sample_hz / 2.0
    wn = min(cutoff_hz / nyq, 0.99)  # must be strictly < 1
    sos = butter(order, wn, btype="low", output="sos")
    return sos


def filter_object_seq(seq: list, cutoff_hz: float, sample_hz: float) -> list:
    """
    Apply a zero-phase Butterworth low-pass filter to a list of arrays (or None).
    Each valid frame must have the same shape.  Gaps (None entries) are filled by
    forward-then-backward fill before filtering, then restored to None afterward.
    """
    if cutoff_hz is None:
        return seq

    valid_mask = np.array([x is not None for x in seq])
    if valid_mask.sum() < 2:
        return seq

    # determine shape from first valid entry
    first = next(x for x in seq if x is not None)
    shape = np.asarray(first).shape
    n = len(seq)

    # fill gaps with nearest valid value for filtering, then mask back out
    filled = np.zeros((n,) + shape, dtype=float)
    last_valid = None
    for i, x in enumerate(seq):
        if x is not None:
            last_valid = np.asarray(x, dtype=float)
        if last_valid is not None:
            filled[i] = last_valid
    # backward fill leading Nones
    first_valid_idx = int(np.argmax(valid_mask))
    if first_valid_idx > 0:
        filled[:first_valid_idx] = filled[first_valid_idx]

    sos = _butter_lowpass_sos(cutoff_hz, sample_hz)
    flat = filled.reshape(n, -1)
    filtered_flat = sosfiltfilt(sos, flat, axis=0)
    filtered = filtered_flat.reshape((n,) + shape)

    return [filtered[i] if valid_mask[i] else None for i in range(n)]


# ── outlier rejection ────────────────────────────────────────────────────────

def reject_raw_pose_outlier_rows(
    rows: list,
    max_pos_speed_m_s: float,
    max_rot_speed_deg_s: float,
    recovery_frames: int = 4,
) -> list:
    """
    Remove raw JSONL pose rows whose implied speed exceeds the threshold, plus
    the next `recovery_frames` rows after each bad frame.  Applied BEFORE SLERP
    resampling so interpolation never touches bad frames.

    The recovery window matters because false ArUco detections typically disturb
    the tracker for several frames after the bad detection; those recovery frames
    look individually plausible but are still biased toward the wrong pose.
    """
    cleaned = []
    prev_ts = None
    prev_pos = None
    prev_q_xyzw = None
    skip_remaining = 0   # countdown of recovery frames still to drop

    for r in rows:
        if skip_remaining > 0:
            skip_remaining -= 1
            continue

        ts = float(r["timestamp"])
        p = np.asarray(r["pose_world"], dtype=float)
        pos = p[:3]
        qw, qx, qy, qz = p[3:]
        q_xyzw = np.array([qx, qy, qz, qw])

        accept = True
        if prev_ts is not None:
            dt = ts - prev_ts
            if dt > 1e-6:
                if np.linalg.norm(pos - prev_pos) / dt > max_pos_speed_m_s:
                    accept = False

                if accept:
                    q_cmp = q_xyzw if np.dot(q_xyzw, prev_q_xyzw) >= 0 else -q_xyzw
                    rel = Rotation.from_quat(prev_q_xyzw).inv() * Rotation.from_quat(q_cmp)
                    if np.rad2deg(rel.magnitude()) / dt > max_rot_speed_deg_s:
                        accept = False

        if accept:
            cleaned.append(r)
            prev_ts = ts
            prev_pos = pos
            prev_q_xyzw = q_xyzw
        else:
            skip_remaining = recovery_frames

    return cleaned


# ── world-frame transform ─────────────────────────────────────────────────────

def to_world_frame(joints_wrist, pose_world_board, T_board_wrist):
    T_world_board = pose_to_matrix(pose_world_board)
    T_world_wrist = T_world_board @ T_board_wrist
    joints_world = transform_points(T_world_wrist, joints_wrist)
    return joints_world, T_world_wrist


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    hands_path      = RAW_RECORDING_DIR / "hands.jsonl"
    ergonomics_path = RAW_RECORDING_DIR / "ergonomics.jsonl"
    poses_path      = RAW_RECORDING_DIR / "poses.jsonl"
    meta_path       = RAW_RECORDING_DIR / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    hands      = group_by_stream(read_jsonl(hands_path))
    ergonomics = group_by_stream(read_jsonl(ergonomics_path))
    poses      = group_by_stream(read_jsonl(poses_path))

    all_timestamps = [
        r["timestamp"]
        for grouped in [hands, ergonomics, poses]
        for stream_rows in grouped.values()
        for r in stream_rows
    ]
    if not all_timestamps:
        raise RuntimeError("No timestamps found in recording.")

    timeline = build_uniform_timeline(min(all_timestamps), max(all_timestamps), TARGET_HZ)

    T_board_wrist_left  = load_calibration_matrix(LEFT_CALIB_JSON)
    T_board_wrist_right = load_calibration_matrix(RIGHT_CALIB_JSON)

    # ── outlier rejection on raw rows (before SLERP) ─────────────────────────
    print(f"Rejecting raw pose outliers (pos>{OUTLIER_MAX_POS_SPEED_M_S:.2f} m/s, rot>{OUTLIER_MAX_ROT_SPEED_DEG_S:.0f} deg/s) …")
    for stream in ("left_board", "right_board", "cube"):
        raw = poses.get(stream, [])
        cleaned = reject_raw_pose_outlier_rows(raw, OUTLIER_MAX_POS_SPEED_M_S, OUTLIER_MAX_ROT_SPEED_DEG_S, OUTLIER_RECOVERY_FRAMES)
        if len(cleaned) < len(raw):
            print(f"  {stream}: removed {len(raw) - len(cleaned)} / {len(raw)} raw frames")
        poses[stream] = cleaned

    # ── resample onto uniform timeline ────────────────────────────────────────
    left_joints_wrist_seq  = [interp_hand_joints(hands.get("left_hand",  []), t) for t in timeline]
    right_joints_wrist_seq = [interp_hand_joints(hands.get("right_hand", []), t) for t in timeline]

    left_ergo_seq  = [interp_angles(ergonomics.get("left_ergonomics",  []), t) for t in timeline]
    right_ergo_seq = [interp_angles(ergonomics.get("right_ergonomics", []), t) for t in timeline]

    left_board_pose_seq  = [interp_pose_slerp(poses.get("left_board",  []), t) for t in timeline]
    right_board_pose_seq = [interp_pose_slerp(poses.get("right_board", []), t) for t in timeline]
    cube_pose_seq        = [interp_pose_slerp(poses.get("cube",        []), t) for t in timeline]

    # ── filter ────────────────────────────────────────────────────────────────
    def filter_pose_full(pose_seq, cutoff_hz):
        # Translation: direct filter
        trans_seq = [p[:3] if p is not None else None for p in pose_seq]
        filtered_trans = filter_object_seq(trans_seq, cutoff_hz, TARGET_HZ)

        # Quaternion [qw,qx,qy,qz]: fix sign flips so the low-pass doesn't
        # average across a hemisphere boundary, then filter and renormalize.
        quats = [p[3:].copy() if p is not None else None for p in pose_seq]
        prev = None
        for i, q in enumerate(quats):
            if q is None:
                continue
            if prev is not None and np.dot(q, prev) < 0:
                quats[i] = -q
            prev = quats[i]
        filtered_quats = filter_object_seq(quats, cutoff_hz, TARGET_HZ)

        result = []
        for i in range(len(pose_seq)):
            if pose_seq[i] is None:
                result.append(None)
            else:
                q = filtered_quats[i] / np.linalg.norm(filtered_quats[i])
                result.append(np.concatenate([filtered_trans[i], q]))
        return result

    if FILTER_HZ_POSES is not None:
        print(f"Applying {FILTER_HZ_POSES} Hz low-pass filter to tag poses …")
        left_board_pose_seq  = filter_pose_full(left_board_pose_seq,  FILTER_HZ_POSES)
        right_board_pose_seq = filter_pose_full(right_board_pose_seq, FILTER_HZ_POSES)
        cube_pose_seq        = filter_pose_full(cube_pose_seq,        FILTER_HZ_POSES)

    if FILTER_HZ_HANDS is not None:
        print(f"Applying {FILTER_HZ_HANDS} Hz low-pass filter to hand joints and ergonomics …")
        left_joints_wrist_seq  = filter_object_seq(left_joints_wrist_seq,  FILTER_HZ_HANDS, TARGET_HZ)
        right_joints_wrist_seq = filter_object_seq(right_joints_wrist_seq, FILTER_HZ_HANDS, TARGET_HZ)
        left_ergo_seq          = filter_object_seq(left_ergo_seq,          FILTER_HZ_HANDS, TARGET_HZ)
        right_ergo_seq         = filter_object_seq(right_ergo_seq,         FILTER_HZ_HANDS, TARGET_HZ)

    # ── world-frame transforms ────────────────────────────────────────────────
    left_joints_world_seq  = []
    right_joints_world_seq = []
    left_wrist_world_seq   = []
    right_wrist_world_seq  = []

    for i in range(len(timeline)):
        lj, lp = left_joints_wrist_seq[i],  left_board_pose_seq[i]
        rj, rp = right_joints_wrist_seq[i], right_board_pose_seq[i]

        if lj is not None and lp is not None:
            lw, lt = to_world_frame(lj, lp, T_board_wrist_left)
        else:
            lw, lt = None, None

        if rj is not None and rp is not None:
            rw, rt = to_world_frame(rj, rp, T_board_wrist_right)
        else:
            rw, rt = None, None

        left_joints_world_seq.append(lw)
        right_joints_world_seq.append(rw)
        left_wrist_world_seq.append(lt)
        right_wrist_world_seq.append(rt)

    # ── save ──────────────────────────────────────────────────────────────────
    def obj(seq):
        return np.array(seq, dtype=object)

    output = {
        "timeline":    np.array(timeline, dtype=float),
        "target_hz":        np.array([TARGET_HZ], dtype=float),
        "filter_hz_poses":           np.array([FILTER_HZ_POSES if FILTER_HZ_POSES is not None else 0.0], dtype=float),
        "filter_hz_hands":           np.array([FILTER_HZ_HANDS if FILTER_HZ_HANDS is not None else 0.0], dtype=float),
        "outlier_max_pos_speed_m_s": np.array([OUTLIER_MAX_POS_SPEED_M_S], dtype=float),
        "outlier_max_rot_speed_deg_s": np.array([OUTLIER_MAX_ROT_SPEED_DEG_S], dtype=float),

        "left_joints_wrist":  obj(left_joints_wrist_seq),
        "right_joints_wrist": obj(right_joints_wrist_seq),

        "left_joints_world":  obj(left_joints_world_seq),
        "right_joints_world": obj(right_joints_world_seq),

        "left_wrist_world":  obj(left_wrist_world_seq),
        "right_wrist_world": obj(right_wrist_world_seq),

        "left_ergonomics":  obj(left_ergo_seq),
        "right_ergonomics": obj(right_ergo_seq),

        "left_board_pose_world":  obj(left_board_pose_seq),
        "right_board_pose_world": obj(right_board_pose_seq),
        "cube_pose_world":        obj(cube_pose_seq),

        "recording_dir": np.array([str(RAW_RECORDING_DIR)], dtype=object),
        "meta_json":     np.array([json.dumps(meta)], dtype=object),
    }

    output_path = RAW_RECORDING_DIR / OUTPUT_FILENAME
    np.savez_compressed(output_path, **output)

    print(f"Saved processed sequence to: {output_path}")
    print(f"Timeline: {len(timeline)} frames at {TARGET_HZ:.1f} Hz")
    print(f"  pose filter: {FILTER_HZ_POSES} Hz  |  hand filter: {FILTER_HZ_HANDS} Hz")


if __name__ == "__main__":
    main()
