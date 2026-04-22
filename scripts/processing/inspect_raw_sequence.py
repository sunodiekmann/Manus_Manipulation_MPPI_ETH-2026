"""
Inspect a raw recording directly from the JSONL event streams.

Input:  recordings/<session>/hands.jsonl, ergonomics.jsonl, poses.jsonl

What it does:
  - Loads the raw, unsynchronised event streams as recorded by record_demo_sequence.py.
  - Plots per-stream timestamps, arrival rates, and raw data values so you can
    quickly sanity-check whether all streams were running and at what frequency.
  - Useful for diagnosing recording issues (missing streams, dropped packets,
    timestamp jitter) before committing to the full processing pipeline.

Run after recording, before build_sequence_dataset.py.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


import sys as _sys
_session = Path(_sys.argv[1]) if len(_sys.argv) > 1 else Path("recordings/statics/session_20260421_150421")
HANDS_PATH      = _session / "hands.jsonl"
ERGONOMICS_PATH = _session / "ergonomics.jsonl"
POSES_PATH      = _session / "poses.jsonl"


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_stream(rows):
    grouped = {}
    for row in rows:
        stream = row.get("stream", "unknown")
        grouped.setdefault(stream, []).append(row)
    for stream in grouped:
        grouped[stream] = sorted(grouped[stream], key=lambda r: r["timestamp"])
    return grouped


def extract_timestamps(rows):
    ts = [r["timestamp"] for r in rows if r.get("timestamp") is not None]
    ts = np.array(ts, dtype=float)
    if len(ts) > 0:
        ts = np.sort(ts)
    return ts


def extract_positions(rows):
    ts = []
    pos = []

    for r in rows:
        t = r.get("timestamp")
        p = r.get("pose_world")
        if t is None or p is None:
            continue

        ts.append(float(t))
        pos.append([float(p[0]), float(p[1]), float(p[2])])

    if len(ts) == 0:
        return np.array([]), np.empty((0, 3))

    ts = np.array(ts, dtype=float)
    pos = np.array(pos, dtype=float)

    order = np.argsort(ts)
    return ts[order], pos[order]


def extract_rotations(rows):
    ts = []
    rots = []

    for r in rows:
        t = r.get("timestamp")
        p = r.get("pose_world")
        if t is None or p is None:
            continue

        qw, qx, qy, qz = p[3:]
        rot = R.from_quat([qx, qy, qz, qw])

        ts.append(float(t))
        rots.append(rot)

    if len(ts) == 0:
        return np.array([]), []

    ts = np.array(ts, dtype=float)
    order = np.argsort(ts)
    ts = ts[order]
    rots = [rots[i] for i in order]
    return ts, rots


def extract_euler_angles(rows, euler_order="xyz", degrees=True):
    ts, rots = extract_rotations(rows)
    if len(ts) == 0:
        return np.array([]), np.empty((0, 3))

    eulers = np.array([rot.as_euler(euler_order, degrees=degrees) for rot in rots], dtype=float)

    if degrees:
        eulers = np.rad2deg(np.unwrap(np.deg2rad(eulers), axis=0))
    else:
        eulers = np.unwrap(eulers, axis=0)

    return ts, eulers


def compute_dt(ts):
    if len(ts) < 2:
        return np.array([]), np.array([])
    dt = np.diff(ts)
    return ts[1:], dt


def compute_speed(ts, pos):
    if len(ts) < 2:
        return np.array([]), np.array([])
    dt = np.diff(ts)
    dp = np.diff(pos, axis=0)
    speed = np.linalg.norm(dp, axis=1) / np.maximum(dt, 1e-9)
    return ts[1:], speed


def compute_orientation_jump_deg(ts, rots):
    if len(ts) < 2:
        return np.array([]), np.array([])
    jump_deg = []
    for i in range(1, len(rots)):
        rel = rots[i - 1].inv() * rots[i]
        jump_deg.append(np.rad2deg(rel.magnitude()))
    return ts[1:], np.array(jump_deg, dtype=float)


def print_stream_summary(name, ts, pos_rows=None, rot_rows=None):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"samples: {len(ts)}")

    if len(ts) >= 2:
        dt = np.diff(ts)
        print(f"mean dt: {dt.mean():.4f} s")
        print(f"std dt : {dt.std():.4f} s")
        print(f"max dt : {dt.max():.4f} s")

    if pos_rows is not None and len(ts) >= 2:
        _, speed = compute_speed(ts, pos_rows)
        if len(speed) > 0:
            print(f"max speed: {speed.max():.4f} m/s")

    if rot_rows is not None and len(ts) >= 2:
        _, jumps = compute_orientation_jump_deg(ts, rot_rows)
        if len(jumps) > 0:
            print(f"max orientation jump: {jumps.max():.2f} deg")


def plot_event_timestamps_by_stream(all_groups):
    fig, ax = plt.subplots(figsize=(12, 6))

    y = 0
    yticks = []
    ylabels = []

    for name, rows in all_groups.items():
        ts = extract_timestamps(rows)
        if len(ts) > 0:
            ax.vlines(ts, y - 0.35, y + 0.35)
        yticks.append(y)
        ylabels.append(name)
        y += 1

    ax.set_title("Event timestamps by stream")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("time [s, unix]")
    ax.set_ylabel("stream")
    ax.grid(True)

    fig.tight_layout()
    return fig


def plot_dt_by_stream(all_groups):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, rows in all_groups.items():
        ts = extract_timestamps(rows)
        t_dt, dt = compute_dt(ts)
        if len(dt) == 0:
            continue
        ax.plot(t_dt - ts[0], dt, label=name)

    ax.set_title("Inter-sample dt by stream")
    ax.set_xlabel("time since first sample [s]")
    ax.set_ylabel("dt [s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_world_position_time_series(pose_groups):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    stream_names = ["left_board", "right_board", "cube"]
    all_ts = []

    for name in stream_names:
        ts, _ = extract_positions(pose_groups.get(name, []))
        if len(ts) > 0:
            all_ts.append(ts)

    if not all_ts:
        return fig

    global_t0 = min(ts[0] for ts in all_ts)

    for name in stream_names:
        ts, pos = extract_positions(pose_groups.get(name, []))
        if len(ts) == 0:
            continue

        trel = ts - global_t0
        axes[0].plot(trel, pos[:, 0], label=name)
        axes[1].plot(trel, pos[:, 1], label=name)
        axes[2].plot(trel, pos[:, 2], label=name)

    axes[0].set_title("World position time series")
    axes[0].set_ylabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[2].set_ylabel("z [m]")
    axes[2].set_xlabel("time since first pose sample [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_world_orientation_time_series(pose_groups, euler_order="xyz"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    stream_names = ["left_board", "right_board", "cube"]
    all_ts = []

    for name in stream_names:
        ts, _ = extract_euler_angles(pose_groups.get(name, []), euler_order=euler_order, degrees=True)
        if len(ts) > 0:
            all_ts.append(ts)

    if not all_ts:
        return fig

    global_t0 = min(ts[0] for ts in all_ts)

    for name in stream_names:
        ts, eulers = extract_euler_angles(pose_groups.get(name, []), euler_order=euler_order, degrees=True)
        if len(ts) == 0:
            continue

        trel = ts - global_t0
        axes[0].plot(trel, eulers[:, 0], label=name)
        axes[1].plot(trel, eulers[:, 1], label=name)
        axes[2].plot(trel, eulers[:, 2], label=name)

    axes[0].set_title(f"World orientation time series (Euler {euler_order})")
    axes[0].set_ylabel(f"{euler_order[0]} [deg]")
    axes[1].set_ylabel(f"{euler_order[1]} [deg]")
    axes[2].set_ylabel(f"{euler_order[2]} [deg]")
    axes[2].set_xlabel("time since first pose sample [s]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_position_speed(pose_groups):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name in ["left_board", "right_board", "cube"]:
        ts, pos = extract_positions(pose_groups.get(name, []))
        if len(ts) < 2:
            continue
        t_speed, speed = compute_speed(ts, pos)
        ax.plot(t_speed - ts[0], speed, label=name)

    ax.set_title("Position speed magnitude")
    ax.set_xlabel("time since first sample [s]")
    ax.set_ylabel("speed [m/s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_orientation_jump(pose_groups):
    fig, ax = plt.subplots(figsize=(12, 5))

    for name in ["left_board", "right_board", "cube"]:
        ts, rots = extract_rotations(pose_groups.get(name, []))
        if len(ts) < 2:
            continue
        t_jump, jumps = compute_orientation_jump_deg(ts, rots)
        ax.plot(t_jump - ts[0], jumps, label=name)

    ax.set_title("Orientation jump magnitude between consecutive samples")
    ax.set_xlabel("time since first sample [s]")
    ax.set_ylabel("jump [deg]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    hand_rows = load_jsonl(HANDS_PATH)
    ergonomics_rows = load_jsonl(ERGONOMICS_PATH)
    pose_rows = load_jsonl(POSES_PATH)

    hand_groups = group_by_stream(hand_rows)
    ergonomics_groups = group_by_stream(ergonomics_rows)
    pose_groups = group_by_stream(pose_rows)

    all_groups = {}
    all_groups.update(hand_groups)
    all_groups.update(ergonomics_groups)
    all_groups.update(pose_groups)

    for name in ["left_board", "right_board", "cube"]:
        ts, pos = extract_positions(pose_groups.get(name, []))
        _, rots = extract_rotations(pose_groups.get(name, []))
        if len(ts) > 0:
            print_stream_summary(name, ts, pos_rows=pos, rot_rows=rots)

    for name in ["left_hand", "right_hand", "left_ergonomics", "right_ergonomics"]:
        ts = extract_timestamps(all_groups.get(name, []))
        if len(ts) > 0:
            print_stream_summary(name, ts)

    plot_event_timestamps_by_stream(all_groups)
    plot_dt_by_stream(all_groups)
    plot_world_position_time_series(pose_groups)
    plot_world_orientation_time_series(pose_groups, euler_order="xyz")
    plot_position_speed(pose_groups)
    plot_orientation_jump(pose_groups)

    plt.show()


if __name__ == "__main__":
    main()