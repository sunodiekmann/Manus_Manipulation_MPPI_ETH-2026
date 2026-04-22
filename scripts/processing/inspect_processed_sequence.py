"""
Inspect a processed sequence (.npz) with interactive plots.

Input:  recordings/<session>/processed_sequence.npz

What it does:
  - Loads a sequence built by build_sequence_dataset.py.
  - Prints a text summary: frame count, timing statistics, max speed, max
    orientation jumps per stream.
  - Plots position trajectories, Euler-angle orientations, speed magnitudes,
    and frame-to-frame orientation jumps for the left/right board poses, cube
    pose, and both wrist transforms.
  - Useful for verifying that the synchronisation, interpolation, and
    wrist→world transform are working correctly before running downstream
    processing (compute_proxy_geometry) or evaluation.

Run after build_sequence_dataset.py.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


import sys as _sys
_arg = Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path("recordings/statics/session_20260421_150421")
INPUT_PATH = _arg if _arg.suffix == ".npz" else _arg / "processed_sequence.npz"


def load_processed(path: Path):
    return np.load(path, allow_pickle=True)


def object_array_to_list(arr):
    return list(arr.tolist())


def valid_mask_from_object_array(arr):
    return np.array([x is not None for x in arr], dtype=bool)


def extract_pose_positions(pose_seq):
    valid = valid_mask_from_object_array(pose_seq)
    pos = np.full((len(pose_seq), 3), np.nan, dtype=float)

    for i, pose in enumerate(object_array_to_list(pose_seq)):
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=float)
        pos[i] = pose[:3]

    return valid, pos


def extract_pose_rotations(pose_seq):
    valid = valid_mask_from_object_array(pose_seq)
    rots = [None] * len(pose_seq)

    for i, pose in enumerate(object_array_to_list(pose_seq)):
        if pose is None:
            continue
        pose = np.asarray(pose, dtype=float)
        qw, qx, qy, qz = pose[3:]
        rots[i] = R.from_quat([qx, qy, qz, qw])

    return valid, rots


def extract_transform_positions(T_seq):
    valid = valid_mask_from_object_array(T_seq)
    pos = np.full((len(T_seq), 3), np.nan, dtype=float)

    for i, T in enumerate(object_array_to_list(T_seq)):
        if T is None:
            continue
        T = np.asarray(T, dtype=float)
        pos[i] = T[:3, 3]

    return valid, pos


def extract_transform_rotations(T_seq):
    valid = valid_mask_from_object_array(T_seq)
    rots = [None] * len(T_seq)

    for i, T in enumerate(object_array_to_list(T_seq)):
        if T is None:
            continue
        T = np.asarray(T, dtype=float)
        rots[i] = R.from_matrix(T[:3, :3])

    return valid, rots


def rotations_to_euler(rots, euler_order="xyz", degrees=True):
    eulers = np.full((len(rots), 3), np.nan, dtype=float)

    for i, rot in enumerate(rots):
        if rot is None:
            continue
        eulers[i] = rot.as_euler(euler_order, degrees=degrees)

    valid_rows = ~np.isnan(eulers[:, 0])
    if np.any(valid_rows):
        if degrees:
            eulers_valid = np.rad2deg(np.unwrap(np.deg2rad(eulers[valid_rows]), axis=0))
        else:
            eulers_valid = np.unwrap(eulers[valid_rows], axis=0)
        eulers[valid_rows] = eulers_valid

    return eulers


def compute_dt(timeline):
    if len(timeline) < 2:
        return np.array([]), np.array([])
    return timeline[1:], np.diff(timeline)


def compute_speed(timeline, pos):
    if len(timeline) < 2:
        return np.array([]), np.array([])

    dt = np.diff(timeline)
    dp = np.diff(pos, axis=0)
    valid = (
        np.isfinite(dp[:, 0]) &
        np.isfinite(dp[:, 1]) &
        np.isfinite(dp[:, 2]) &
        (dt > 0)
    )

    speed = np.full(len(dt), np.nan, dtype=float)
    speed[valid] = np.linalg.norm(dp[valid], axis=1) / dt[valid]
    return timeline[1:], speed


def compute_orientation_jump_deg(timeline, rots):
    if len(timeline) < 2:
        return np.array([]), np.array([])

    jumps = np.full(len(timeline) - 1, np.nan, dtype=float)

    for i in range(1, len(rots)):
        if rots[i - 1] is None or rots[i] is None:
            continue
        rel = rots[i - 1].inv() * rots[i]
        jumps[i - 1] = np.rad2deg(rel.magnitude())

    return timeline[1:], jumps


def print_summary(name, timeline, pos=None, rots=None):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"frames: {len(timeline)}")

    if len(timeline) >= 2:
        dt = np.diff(timeline)
        print(f"mean dt: {dt.mean():.4f} s")
        print(f"std dt : {dt.std():.4f} s")
        print(f"max dt : {dt.max():.4f} s")

    if pos is not None:
        _, speed = compute_speed(timeline, pos)
        finite = np.isfinite(speed)
        if np.any(finite):
            print(f"max speed: {np.nanmax(speed):.4f} m/s")

    if rots is not None:
        _, jumps = compute_orientation_jump_deg(timeline, rots)
        finite = np.isfinite(jumps)
        if np.any(finite):
            print(f"max orientation jump: {np.nanmax(jumps):.2f} deg")


def plot_dt(timeline):
    fig, ax = plt.subplots(figsize=(10, 4))
    t_dt, dt = compute_dt(timeline)
    if len(dt) > 0:
        ax.plot(t_dt - timeline[0], dt)
    ax.set_title("Processed timeline dt")
    ax.set_xlabel("time since start [s]")
    ax.set_ylabel("dt [s]")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_positions(timeline, sequences, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for name, pos in sequences.items():
        axes[0].plot(timeline - timeline[0], pos[:, 0], label=name)
        axes[1].plot(timeline - timeline[0], pos[:, 1], label=name)
        axes[2].plot(timeline - timeline[0], pos[:, 2], label=name)

    axes[0].set_title(title)
    axes[0].set_ylabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[2].set_ylabel("z [m]")
    axes[2].set_xlabel("time since start [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_eulers(timeline, sequences, title, euler_order="xyz"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for name, eulers in sequences.items():
        axes[0].plot(timeline - timeline[0], eulers[:, 0], label=name)
        axes[1].plot(timeline - timeline[0], eulers[:, 1], label=name)
        axes[2].plot(timeline - timeline[0], eulers[:, 2], label=name)

    axes[0].set_title(f"{title} (Euler {euler_order})")
    axes[0].set_ylabel(f"{euler_order[0]} [deg]")
    axes[1].set_ylabel(f"{euler_order[1]} [deg]")
    axes[2].set_ylabel(f"{euler_order[2]} [deg]")
    axes[2].set_xlabel("time since start [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_speed(timeline, sequences, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, pos in sequences.items():
        t_speed, speed = compute_speed(timeline, pos)
        ax.plot(t_speed - timeline[0], speed, label=name)

    ax.set_title(title)
    ax.set_xlabel("time since start [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_orientation_jump(timeline, sequences, title):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, rots in sequences.items():
        t_jump, jumps = compute_orientation_jump_deg(timeline, rots)
        ax.plot(t_jump - timeline[0], jumps, label=name)

    ax.set_title(title)
    ax.set_xlabel("time since start [s]")
    ax.set_ylabel("jump [deg]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    data = load_processed(INPUT_PATH)

    timeline = np.asarray(data["timeline"], dtype=float)

    left_board_pose_world = data["left_board_pose_world"]
    right_board_pose_world = data["right_board_pose_world"]
    cube_pose_world = data["cube_pose_world"]

    left_wrist_world = data["left_wrist_world"]
    right_wrist_world = data["right_wrist_world"]

    # Pose sequences
    _, left_board_pos = extract_pose_positions(left_board_pose_world)
    _, right_board_pos = extract_pose_positions(right_board_pose_world)
    _, cube_pos = extract_pose_positions(cube_pose_world)

    _, left_board_rots = extract_pose_rotations(left_board_pose_world)
    _, right_board_rots = extract_pose_rotations(right_board_pose_world)
    _, cube_rots = extract_pose_rotations(cube_pose_world)

    left_board_eulers = rotations_to_euler(left_board_rots, euler_order="xyz", degrees=True)
    right_board_eulers = rotations_to_euler(right_board_rots, euler_order="xyz", degrees=True)
    cube_eulers = rotations_to_euler(cube_rots, euler_order="xyz", degrees=True)

    # Wrist transforms
    _, left_wrist_pos = extract_transform_positions(left_wrist_world)
    _, right_wrist_pos = extract_transform_positions(right_wrist_world)

    _, left_wrist_rots = extract_transform_rotations(left_wrist_world)
    _, right_wrist_rots = extract_transform_rotations(right_wrist_world)

    left_wrist_eulers = rotations_to_euler(left_wrist_rots, euler_order="xyz", degrees=True)
    right_wrist_eulers = rotations_to_euler(right_wrist_rots, euler_order="xyz", degrees=True)

    # Summary
    print_summary("left_board_pose_world", timeline, left_board_pos, left_board_rots)
    print_summary("right_board_pose_world", timeline, right_board_pos, right_board_rots)
    print_summary("cube_pose_world", timeline, cube_pos, cube_rots)
    print_summary("left_wrist_world", timeline, left_wrist_pos, left_wrist_rots)
    print_summary("right_wrist_world", timeline, right_wrist_pos, right_wrist_rots)

    meta_json = data.get("meta_json", None)
    if meta_json is not None and len(meta_json) > 0:
        try:
            meta = json.loads(meta_json[0])
            print("\nMeta")
            print("----")
            print(json.dumps(meta, indent=2))
        except Exception:
            pass

    # Plots
    plot_dt(timeline)

    plot_positions(
        timeline,
        {
            "left_board": left_board_pos,
            "right_board": right_board_pos,
            "cube": cube_pos,
        },
        title="Processed pose positions in world frame",
    )

    plot_eulers(
        timeline,
        {
            "left_board": left_board_eulers,
            "right_board": right_board_eulers,
            "cube": cube_eulers,
        },
        title="Processed pose orientations in world frame",
        euler_order="xyz",
    )

    plot_speed(
        timeline,
        {
            "left_board": left_board_pos,
            "right_board": right_board_pos,
            "cube": cube_pos,
        },
        title="Processed pose speed magnitude",
    )

    plot_orientation_jump(
        timeline,
        {
            "left_board": left_board_rots,
            "right_board": right_board_rots,
            "cube": cube_rots,
        },
        title="Processed pose orientation jump magnitude",
    )

    plot_positions(
        timeline,
        {
            "left_wrist": left_wrist_pos,
            "right_wrist": right_wrist_pos,
        },
        title="Processed wrist positions in world frame",
    )

    plot_eulers(
        timeline,
        {
            "left_wrist": left_wrist_eulers,
            "right_wrist": right_wrist_eulers,
        },
        title="Processed wrist orientations in world frame",
        euler_order="xyz",
    )

    plot_speed(
        timeline,
        {
            "left_wrist": left_wrist_pos,
            "right_wrist": right_wrist_pos,
        },
        title="Processed wrist speed magnitude",
    )

    plot_orientation_jump(
        timeline,
        {
            "left_wrist": left_wrist_rots,
            "right_wrist": right_wrist_rots,
        },
        title="Processed wrist orientation jump magnitude",
    )

    plt.show()


if __name__ == "__main__":
    main()