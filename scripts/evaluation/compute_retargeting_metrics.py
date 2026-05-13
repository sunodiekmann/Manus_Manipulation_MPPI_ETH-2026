"""
Compute retargeting quality metrics for a recorded session.

For each frame the robot's fingertip positions (from forward kinematics on the
retargeted qpos) are compared to the human's fingertip positions, both expressed
in the cube frame.  Using the cube frame removes any global wrist-pose error and
isolates how well the relative finger configuration is preserved.

Metric per finger per frame:
    error [mm] = ||robot_fingertip_cube - human_fingertip_cube||

Inputs:
    recordings/<session>/retargeted_sequence_cube_relative.npz  (retargeted_qpos)
    recordings/<session>/processed_sequence_with_proxy.npz      (human fingertips + cube)

Output:
    Console: per-finger mean ± std, overall mean ± std
    Plots:
        1. Per-finger error time series
        2. Per-finger mean ± std bar chart

Usage:
    python -m scripts.evaluation.compute_retargeting_metrics recordings/<session>
    python -m scripts.evaluation.compute_retargeting_metrics recordings/          # all sessions
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from scripts.utils.transforms import pose_to_matrix

MODEL_PATH = "/home/suno/robohand2/assets/output_mocap.xml"
FINGERS = ["thumb", "index", "middle", "ring", "little"]
FINGER_GEOMS = [f"{f}_tip" for f in FINGERS]


# ── data loading ──────────────────────────────────────────────────────────────

def load_session(session_dir: Path):
    retargeted_path = session_dir / "retargeted_sequence_cube_relative.npz"
    proxy_path      = session_dir / "processed_sequence_with_proxy.npz"

    if not retargeted_path.exists():
        raise FileNotFoundError(f"Missing: {retargeted_path}\nRun retarget_sequence_cube_relative first.")
    if not proxy_path.exists():
        raise FileNotFoundError(f"Missing: {proxy_path}\nRun compute_proxy_geometry first.")

    ret = np.load(retargeted_path)
    proxy = np.load(proxy_path, allow_pickle=True)
    return ret, proxy


# ── forward kinematics ────────────────────────────────────────────────────────

def get_robot_fingertips_world(model, data, qpos) -> np.ndarray:
    """Set qpos and return (5, 3) robot fingertip positions in world frame."""
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    return np.array([data.geom(g).xpos.copy() for g in FINGER_GEOMS])


# ── per-session metrics ───────────────────────────────────────────────────────

def compute_metrics(session_dir: Path):
    ret, proxy = load_session(session_dir)

    timeline         = np.asarray(ret["timeline"], dtype=float)
    retargeted_qpos  = ret["retargeted_qpos"]          # (N, nq)
    cube_pose_world  = proxy["cube_pose_world"]         # object array length N
    human_tips_cube  = proxy["right_fingertips_cube"]   # object array length N, each (5,3) or None

    N = len(timeline)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    errors_mm = np.full((N, 5), np.nan)   # per frame, per finger

    for i in range(N):
        cube_pose = cube_pose_world[i]
        human_tips = human_tips_cube[i]

        if cube_pose is None or human_tips is None:
            continue
        if np.any(np.isnan(retargeted_qpos[i])):
            continue

        T_world_cube = pose_to_matrix(np.asarray(cube_pose, dtype=float))
        T_cube_world = np.linalg.inv(T_world_cube)

        robot_tips_world = get_robot_fingertips_world(model, data, retargeted_qpos[i])

        # transform robot tips to cube frame
        ones = np.ones((5, 1))
        robot_tips_cube = (T_cube_world @ np.hstack([robot_tips_world, ones]).T).T[:, :3]

        human_tips_arr = np.asarray(human_tips, dtype=float)

        errors_mm[i] = np.linalg.norm(robot_tips_cube - human_tips_arr, axis=1) * 1000.0

    return timeline, errors_mm


# ── printing ──────────────────────────────────────────────────────────────────

def print_summary(session_dir: Path, errors_mm: np.ndarray):
    print(f"\n{session_dir.name}")
    print("─" * 52)
    print(f"  {'finger':<10}  {'mean':>8}  {'std':>8}  {'max':>8}  mm")
    print(f"  {'──────':<10}  {'────':>8}  {'───':>8}  {'───':>8}")

    all_valid = []
    for fi, name in enumerate(FINGERS):
        vals = errors_mm[:, fi]
        valid = vals[np.isfinite(vals)]
        if len(valid) == 0:
            print(f"  {name:<10}  {'n/a':>8}")
            continue
        print(f"  {name:<10}  {valid.mean():8.2f}  {valid.std():8.2f}  {valid.max():8.2f}")
        all_valid.append(valid)

    if all_valid:
        combined = np.concatenate(all_valid)
        print(f"  {'─'*10}  {'────────':>8}  {'────────':>8}  {'────────':>8}")
        print(f"  {'overall':<10}  {combined.mean():8.2f}  {combined.std():8.2f}  {combined.max():8.2f}")


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]


def plot_metrics(session_dir: Path, timeline: np.ndarray, errors_mm: np.ndarray):
    t = timeline - timeline[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Retargeting metrics — {session_dir.name}", fontsize=12)

    # ── time series ───────────────────────────────────────────────────────────
    ax = axes[0]
    for fi, (name, color) in enumerate(zip(FINGERS, COLORS)):
        vals = errors_mm[:, fi].copy()
        vals[~np.isfinite(vals)] = np.nan
        ax.plot(t, vals, label=name, color=color, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("fingertip error [mm]")
    ax.set_title("Per-finger position error in cube frame (frame by frame)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.4)

    # ── bar chart: mean ± std ─────────────────────────────────────────────────
    ax = axes[1]
    means, stds = [], []
    for fi in range(5):
        valid = errors_mm[:, fi]
        valid = valid[np.isfinite(valid)]
        means.append(valid.mean() if len(valid) else np.nan)
        stds.append(valid.std()  if len(valid) else 0.0)

    x = np.arange(5)
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLORS, alpha=0.8, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(FINGERS)
    ax.set_ylabel("mean ± std error [mm]")
    ax.set_title("Per-finger mean ± std")
    ax.grid(True, axis="y", alpha=0.4)

    # annotate bar tops
    for bar, m, s in zip(bars, means, stds):
        if np.isfinite(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.3,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def find_sessions(root: Path):
    return sorted(p.parent for p in root.rglob("retargeted_sequence_cube_relative.npz"))


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m scripts.evaluation.compute_retargeting_metrics <session_or_dir>")
        sys.exit(1)

    root = Path(args[0])
    if (root / "retargeted_sequence_cube_relative.npz").exists():
        sessions = [root]
    else:
        sessions = find_sessions(root)

    if not sessions:
        print(f"No retargeted sessions found under {root}")
        sys.exit(1)

    for session_dir in sessions:
        try:
            timeline, errors_mm = compute_metrics(session_dir)
        except FileNotFoundError as e:
            print(f"[skip] {session_dir.name}: {e}")
            continue

        print_summary(session_dir, errors_mm)
        plot_metrics(session_dir, timeline, errors_mm)

    plt.show()


if __name__ == "__main__":
    main()
