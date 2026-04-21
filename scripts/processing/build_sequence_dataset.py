import json
from pathlib import Path

import numpy as np

from scripts.utils.transforms import pose_to_matrix, transform_points
from scripts.utils.calibration_io import load_calibration_matrix


RAW_RECORDING_DIR = Path("recordings/statics/session_20260421_150421")
OUTPUT_FILENAME = "processed_sequence.npz"

import sys as _sys
if len(_sys.argv) > 1:
    RAW_RECORDING_DIR = Path(_sys.argv[1])

LEFT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/left_wrist2board_calibration.json"
RIGHT_CALIB_JSON = "src/tag-pose-estimation/config/calibration_gloves/right_wrist2board_calibration.json"

TARGET_HZ = 20.0


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


def nearest_sample(rows, t):
    if len(rows) == 0:
        return None

    timestamps = np.array([r["timestamp"] for r in rows], dtype=float)
    idx = int(np.argmin(np.abs(timestamps - t)))
    return rows[idx]


def interp_vector(rows, key, t):
    """
    Linear interpolation for vector-valued entries like:
    - angles_rad
    - pose_world
    - joints_wrist (flattened per joint)
    """
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

    t0 = timestamps[left_idx]
    t1 = timestamps[right_idx]

    x0 = np.array(rows[left_idx][key], dtype=float)
    x1 = np.array(rows[right_idx][key], dtype=float)

    alpha = (t - t0) / (t1 - t0)
    return (1.0 - alpha) * x0 + alpha * x1


def interp_hand_joints(rows, t):
    x = interp_vector(rows, "joints_wrist", t)
    if x is None:
        return None
    return np.array(x, dtype=float)


def interp_pose(rows, t):
    """
    For now: simple linear interpolation on the 7D pose vector [x, y, z, qw, qx, qy, qz].
    This is acceptable for a first simple version.
    Later you may want quaternion slerp.
    """
    x = interp_vector(rows, "pose_world", t)
    if x is None:
        return None
    return np.array(x, dtype=float)


def interp_angles(rows, t):
    x = interp_vector(rows, "angles_rad", t)
    if x is None:
        return None
    return np.array(x, dtype=float)


def to_world_frame(joints_wrist, pose_world_board, T_board_wrist):
    """
    joints_wrist: (25, 3)
    pose_world_board: 7D pose [x, y, z, qw, qx, qy, qz]
    T_board_wrist: 4x4 transform wrist -> board
    """
    T_world_board = pose_to_matrix(pose_world_board)
    T_world_wrist = T_world_board @ T_board_wrist
    joints_world = transform_points(T_world_wrist, joints_wrist)
    return joints_world, T_world_wrist


def save_processed_sequence(output_path: Path, data: dict):
    np.savez_compressed(output_path, **data)


def main():
    hands_path = RAW_RECORDING_DIR / "hands.jsonl"
    ergonomics_path = RAW_RECORDING_DIR / "ergonomics.jsonl"
    poses_path = RAW_RECORDING_DIR / "poses.jsonl"
    meta_path = RAW_RECORDING_DIR / "meta.json"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    hand_rows = read_jsonl(hands_path)
    ergonomics_rows = read_jsonl(ergonomics_path)
    pose_rows = read_jsonl(poses_path)

    hands = group_by_stream(hand_rows)
    ergonomics = group_by_stream(ergonomics_rows)
    poses = group_by_stream(pose_rows)

    all_timestamps = []

    for grouped in [hands, ergonomics, poses]:
        for stream_rows in grouped.values():
            all_timestamps.extend([r["timestamp"] for r in stream_rows])

    if len(all_timestamps) == 0:
        raise RuntimeError("No timestamps found in recording.")

    start_t = min(all_timestamps)
    end_t = max(all_timestamps)

    timeline = build_uniform_timeline(start_t, end_t, TARGET_HZ)

    T_board_wrist_left = load_calibration_matrix(LEFT_CALIB_JSON)
    T_board_wrist_right = load_calibration_matrix(RIGHT_CALIB_JSON)

    left_joints_wrist_seq = []
    right_joints_wrist_seq = []

    left_joints_world_seq = []
    right_joints_world_seq = []

    left_wrist_world_seq = []
    right_wrist_world_seq = []

    left_ergonomics_seq = []
    right_ergonomics_seq = []

    left_board_pose_seq = []
    right_board_pose_seq = []
    cube_pose_seq = []

    for t in timeline:
        left_joints_wrist = interp_hand_joints(hands.get("left_hand", []), t)
        right_joints_wrist = interp_hand_joints(hands.get("right_hand", []), t)

        left_ergo = interp_angles(ergonomics.get("left_ergonomics", []), t)
        right_ergo = interp_angles(ergonomics.get("right_ergonomics", []), t)

        left_board_pose = interp_pose(poses.get("left_board", []), t)
        right_board_pose = interp_pose(poses.get("right_board", []), t)
        cube_pose = interp_pose(poses.get("cube", []), t)

        left_joints_world = None
        right_joints_world = None
        T_world_wrist_left = None
        T_world_wrist_right = None

        if left_joints_wrist is not None and left_board_pose is not None:
            left_joints_world, T_world_wrist_left = to_world_frame(
                left_joints_wrist,
                left_board_pose,
                T_board_wrist_left,
            )

        if right_joints_wrist is not None and right_board_pose is not None:
            right_joints_world, T_world_wrist_right = to_world_frame(
                right_joints_wrist,
                right_board_pose,
                T_board_wrist_right,
            )

        left_joints_wrist_seq.append(left_joints_wrist)
        right_joints_wrist_seq.append(right_joints_wrist)

        left_joints_world_seq.append(left_joints_world)
        right_joints_world_seq.append(right_joints_world)

        left_wrist_world_seq.append(T_world_wrist_left)
        right_wrist_world_seq.append(T_world_wrist_right)

        left_ergonomics_seq.append(left_ergo)
        right_ergonomics_seq.append(right_ergo)

        left_board_pose_seq.append(left_board_pose)
        right_board_pose_seq.append(right_board_pose)
        cube_pose_seq.append(cube_pose)

    def stack_object_array(seq):
        return np.array(seq, dtype=object)

    output = {
        "timeline": np.array(timeline, dtype=float),
        "target_hz": np.array([TARGET_HZ], dtype=float),

        "left_joints_wrist": stack_object_array(left_joints_wrist_seq),
        "right_joints_wrist": stack_object_array(right_joints_wrist_seq),

        "left_joints_world": stack_object_array(left_joints_world_seq),
        "right_joints_world": stack_object_array(right_joints_world_seq),

        "left_wrist_world": stack_object_array(left_wrist_world_seq),
        "right_wrist_world": stack_object_array(right_wrist_world_seq),

        "left_ergonomics": stack_object_array(left_ergonomics_seq),
        "right_ergonomics": stack_object_array(right_ergonomics_seq),

        "left_board_pose_world": stack_object_array(left_board_pose_seq),
        "right_board_pose_world": stack_object_array(right_board_pose_seq),
        "cube_pose_world": stack_object_array(cube_pose_seq),

        "recording_dir": np.array([str(RAW_RECORDING_DIR)], dtype=object),
        "meta_json": np.array([json.dumps(meta)], dtype=object),
    }

    output_path = RAW_RECORDING_DIR / OUTPUT_FILENAME
    save_processed_sequence(output_path, output)

    print(f"Saved processed sequence to: {output_path}")
    print(f"Timeline length: {len(timeline)} frames at {TARGET_HZ:.1f} Hz")


if __name__ == "__main__":
    main()