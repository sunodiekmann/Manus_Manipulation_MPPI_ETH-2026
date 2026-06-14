"""Replay a recorded session as a 3D matplotlib animation.

Two modes
---------
processed (default)
    Loads processed_sequence_with_proxy.npz (or processed_sequence.npz).
    Shows the resampled, filtered, calibrated data.

raw  (--raw)
    Loads the original hands.jsonl and poses.jsonl directly.
    Steps through time at PLAYBACK_HZ, showing the most recent sample from
    each stream at every tick.  Useful for inspecting data before processing
    and comparing the effect of filtering / outlier rejection.

Usage:
    python -m scripts.visualization.inspect_capture recordings/<session>
    python -m scripts.visualization.inspect_capture recordings/<session> --raw
    python -m scripts.visualization.inspect_capture recordings/<session> --speed 2.0
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scripts.utils.config import AXIS_LEN, CUBE_SIZE, LEFT_CALIB_JSON, RIGHT_CALIB_JSON
from scripts.utils.calibration_io import load_calibration_matrix
from scripts.utils.transforms import pose_to_matrix, transform_points
from scripts.visualization.hand_plotting import (
    clear_skeleton,
    create_hand_artists,
    update_skeleton,
)
from scripts.visualization.frame_plotting import (
    create_frame_artists,
    update_frame_artists,
)
from scripts.visualization.object_plotting import (
    create_cube_artist,
    update_cube_mesh,
)

PLAYBACK_HZ = 30.0


# ── shared plot setup ─────────────────────────────────────────────────────────

def _make_figure(title: str):
    plt.ion()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.1, 0.6)
    ax.view_init(elev=30, azim=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax


def _make_artists(ax):
    left_scatter,  left_lines,  left_texts  = create_hand_artists(ax, color="blue",  label="Left")
    right_scatter, right_lines, right_texts = create_hand_artists(ax, color="red",   label="Right")
    cube_poly   = create_cube_artist(ax, alpha=0.25, facecolor="cyan",  edgecolor="k")
    left_frame  = create_frame_artists(ax, axis_len=AXIS_LEN)
    right_frame = create_frame_artists(ax, axis_len=AXIS_LEN)
    cube_frame  = create_frame_artists(ax, axis_len=AXIS_LEN)
    frame_text  = ax.text2D(0.02, 0.97, "", transform=ax.transAxes,
                            fontsize=9, verticalalignment="top")
    ax.legend(loc="upper right")
    return (
        (left_scatter, left_lines, left_texts),
        (right_scatter, right_lines, right_texts),
        cube_poly, left_frame, right_frame, cube_frame, frame_text,
    )


# ── processed replay ──────────────────────────────────────────────────────────

def replay(session_dir: Path, speed: float = 1.0):
    for name in ("processed_sequence_with_proxy.npz", "processed_sequence.npz"):
        p = session_dir / name
        if p.exists():
            npz = np.load(p, allow_pickle=True)
            break
    else:
        raise FileNotFoundError(f"No processed data in {session_dir}")

    timeline           = np.asarray(npz["timeline"], dtype=float)
    left_joints_world  = npz["left_joints_world"]
    right_joints_world = npz["right_joints_world"]
    left_wrist_world   = npz["left_wrist_world"]
    right_wrist_world  = npz["right_wrist_world"]
    cube_pose_world    = npz.get("cube_pose_world",
                                  np.array([None] * len(timeline), dtype=object))

    N = len(timeline)
    dt_per_frame = float(np.diff(timeline).mean()) if N > 1 else 1.0 / PLAYBACK_HZ

    fig, ax = _make_figure(f"{session_dir.name}  (processed,  ×{speed})")
    (ls, ll, lt), (rs, rl, rt), cube_poly, lf, rf, cf, ftxt = _make_artists(ax)

    sleep_dt = dt_per_frame / speed
    print(f"  Replaying {N} frames  dt={dt_per_frame*1000:.1f} ms  ×{speed} — close window to stop")

    i = 0
    while plt.fignum_exists(fig.number):
        t0 = time.perf_counter()

        lj = left_joints_world[i]
        rj = right_joints_world[i]
        lw = left_wrist_world[i]
        rw = right_wrist_world[i]
        cp = cube_pose_world[i]

        if lj is not None:
            update_skeleton(np.asarray(lj, dtype=float), ls, ll, lt)
        else:
            clear_skeleton(ls, ll, lt)

        if rj is not None:
            update_skeleton(np.asarray(rj, dtype=float), rs, rl, rt)
        else:
            clear_skeleton(rs, rl, rt)

        if lw is not None:
            update_frame_artists(lf, np.asarray(lw, dtype=float), "L wrist")
        if rw is not None:
            update_frame_artists(rf, np.asarray(rw, dtype=float), "R wrist")

        if cp is not None:
            T_world_cube = pose_to_matrix(np.asarray(cp, dtype=float))
            update_cube_mesh(cube_poly, T_world_cube, CUBE_SIZE)
            update_frame_artists(cf, T_world_cube, "Cube")

        ftxt.set_text(f"frame {i+1}/{N}  t={timeline[i]-timeline[0]:.2f}s")
        plt.pause(0.001)

        elapsed = time.perf_counter() - t0
        remaining = sleep_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        i = (i + 1) % N

    plt.ioff()


# ── raw replay ────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_stream(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("stream", "unknown"), []).append(row)
    for s in grouped:
        grouped[s].sort(key=lambda r: r["timestamp"])
    return grouped


def _latest_before(rows, t):
    """Return the most recent row with timestamp <= t, or None."""
    if not rows:
        return None
    ts = np.array([r["timestamp"] for r in rows])
    idx = int(np.searchsorted(ts, t, side="right")) - 1
    return rows[idx] if idx >= 0 else None


def replay_raw(session_dir: Path, speed: float = 1.0):
    hand_rows  = _load_jsonl(session_dir / "hands.jsonl")
    pose_rows  = _load_jsonl(session_dir / "poses.jsonl")

    if not hand_rows and not pose_rows:
        raise FileNotFoundError(f"No raw data in {session_dir}")

    hand_groups = _group_by_stream(hand_rows)
    pose_groups = _group_by_stream(pose_rows)

    T_board_wrist_left  = load_calibration_matrix(LEFT_CALIB_JSON)
    T_board_wrist_right = load_calibration_matrix(RIGHT_CALIB_JSON)

    all_ts = [r["timestamp"] for r in hand_rows + pose_rows]
    t_start, t_end = min(all_ts), max(all_ts)
    dt = 1.0 / PLAYBACK_HZ
    timeline = np.arange(t_start, t_end, dt)
    N = len(timeline)

    fig, ax = _make_figure(f"{session_dir.name}  (raw,  ×{speed})")
    (ls, ll, lt), (rs, rl, rt), cube_poly, lf, rf, cf, ftxt = _make_artists(ax)

    sleep_dt = (dt / speed)
    print(f"  Raw replay  {N} ticks  ×{speed} — close window to stop")

    left_hand_rows   = hand_groups.get("left_hand",  [])
    right_hand_rows  = hand_groups.get("right_hand", [])
    left_board_rows  = pose_groups.get("left_board", [])
    right_board_rows = pose_groups.get("right_board", [])
    cube_rows        = pose_groups.get("cube", [])

    for i, t in enumerate(timeline):
        if not plt.fignum_exists(fig.number):
            break
        t0 = time.perf_counter()

        l_hand_row  = _latest_before(left_hand_rows,  t)
        l_board_row = _latest_before(left_board_rows, t)
        if l_hand_row and l_board_row:
            pose = np.asarray(l_board_row["pose_world"], dtype=float)
            qw, qx, qy, qz = pose[3:]
            T_world_board = pose_to_matrix([*pose[:3], qw, qx, qy, qz])
            T_world_wrist = T_world_board @ T_board_wrist_left
            joints = transform_points(T_world_wrist,
                                      np.asarray(l_hand_row["joints_wrist"], dtype=float))
            update_skeleton(joints, ls, ll, lt)
            update_frame_artists(lf, T_world_wrist, "L wrist")
        else:
            clear_skeleton(ls, ll, lt)

        r_hand_row  = _latest_before(right_hand_rows,  t)
        r_board_row = _latest_before(right_board_rows, t)
        if r_hand_row and r_board_row:
            pose = np.asarray(r_board_row["pose_world"], dtype=float)
            qw, qx, qy, qz = pose[3:]
            T_world_board = pose_to_matrix([*pose[:3], qw, qx, qy, qz])
            T_world_wrist = T_world_board @ T_board_wrist_right
            joints = transform_points(T_world_wrist,
                                      np.asarray(r_hand_row["joints_wrist"], dtype=float))
            update_skeleton(joints, rs, rl, rt)
            update_frame_artists(rf, T_world_wrist, "R wrist")
        else:
            clear_skeleton(rs, rl, rt)

        cube_row = _latest_before(cube_rows, t)
        if cube_row:
            pose = np.asarray(cube_row["pose_world"], dtype=float)
            qw, qx, qy, qz = pose[3:]
            T_world_cube = pose_to_matrix([*pose[:3], qw, qx, qy, qz])
            update_cube_mesh(cube_poly, T_world_cube, CUBE_SIZE)
            update_frame_artists(cf, T_world_cube, "Cube")

        ftxt.set_text(f"tick {i+1}/{N}  t={t-t_start:.2f}s")
        plt.pause(0.001)

        elapsed = time.perf_counter() - t0
        remaining = sleep_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    plt.ioff()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    raw   = "--raw"   in args
    speed = 1.0
    paths = []

    skip_next = False
    for k, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--speed" and k + 1 < len(args):
            speed = float(args[k + 1])
            skip_next = True
        elif a.startswith("--speed="):
            speed = float(a.split("=", 1)[1])
        elif not a.startswith("--"):
            paths.append(Path(a))

    if not paths:
        print("Usage: python -m scripts.visualization.inspect_capture <session_dir> [--raw] [--speed N]")
        sys.exit(1)

    for p in paths:
        if raw:
            replay_raw(p, speed=speed)
        else:
            replay(p, speed=speed)


if __name__ == "__main__":
    main()
