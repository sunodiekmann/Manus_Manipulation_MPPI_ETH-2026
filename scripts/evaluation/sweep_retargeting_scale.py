"""
Sweep TARGET_SCALE for the baseline retargeting and report per-scale metrics.

For each scale value the script re-retargets all provided sessions, computes
per-finger fingertip error in the cube frame, and prints a compact summary
table so the best scale can be chosen at a glance.

Temporary retargeted files are saved as:
    recordings/<session>/retargeted_sweep_s<scale>.npz
and removed after the sweep unless --keep is passed.

Usage:
    python -m scripts.evaluation.sweep_retargeting_scale recordings/cube_yaw recordings/cube_pitch recordings/cube_roll
    python -m scripts.evaluation.sweep_retargeting_scale recordings/ --scales 1.0 1.1 1.2 1.3 1.4
    python -m scripts.evaluation.sweep_retargeting_scale recordings/ --keep
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from scripts.retargeting.retarget_sequence_baseline import retarget_offline
from scripts.evaluation.compute_retargeting_metrics import compute_metrics, FINGERS
from scripts.utils.transforms import pose_to_matrix

DEFAULT_SCALES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def _output_name(scale: float) -> str:
    return f"retargeted_sweep_s{scale:.2f}.npz"


def _run_one(session_dir: Path, scale: float) -> float:
    """Retarget and return overall mean error [mm]. Returns nan on failure."""
    name = _output_name(scale)
    try:
        retarget_offline(session_dir, show_viewer=False, force=True,
                         scale=scale, output_name=name)
    except Exception as e:
        print(f"    [retarget error] {e}")
        return np.nan

    npz_path = session_dir / name
    if not npz_path.exists():
        return np.nan

    proxy_path = session_dir / "processed_sequence_with_proxy.npz"
    if not proxy_path.exists():
        return np.nan

    try:
        _, errors_mm = _compute_metrics_for_file(session_dir, npz_path, proxy_path)
        valid = errors_mm[np.isfinite(errors_mm)]
        return float(valid.mean()) if len(valid) else np.nan
    except Exception as e:
        print(f"    [metrics error] {e}")
        return np.nan


def _compute_metrics_for_file(session_dir, retargeted_path, proxy_path):
    """Like compute_metrics but uses an explicit retargeted file path."""
    from scripts.evaluation.compute_retargeting_metrics import (
        MODEL_PATH, FINGER_GEOMS, get_robot_fingertips_world,
    )

    ret   = np.load(retargeted_path)
    proxy = np.load(proxy_path, allow_pickle=True)

    timeline        = np.asarray(ret["timeline"], dtype=float)
    retargeted_qpos = ret["retargeted_qpos"]
    cube_pose_world = proxy["cube_pose_world"]
    human_tips_cube = proxy["right_fingertips_cube"]
    N = len(timeline)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    errors_mm = np.full((N, 5), np.nan)
    for i in range(N):
        cube_pose  = cube_pose_world[i]
        human_tips = human_tips_cube[i]
        if cube_pose is None or human_tips is None:
            continue
        if np.any(np.isnan(retargeted_qpos[i])):
            continue

        T_world_cube = pose_to_matrix(np.asarray(cube_pose, dtype=float))
        T_cube_world = np.linalg.inv(T_world_cube)

        robot_tips_world = get_robot_fingertips_world(model, data, retargeted_qpos[i])
        ones = np.ones((5, 1))
        robot_tips_cube  = (T_cube_world @ np.hstack([robot_tips_world, ones]).T).T[:, :3]
        errors_mm[i]     = np.linalg.norm(robot_tips_cube - np.asarray(human_tips, dtype=float), axis=1) * 1000.0

    return timeline, errors_mm


def main():
    args = sys.argv[1:]
    keep = "--keep" in args
    args = [a for a in args if a != "--keep"]

    scales = DEFAULT_SCALES
    if "--scales" in args:
        idx = args.index("--scales")
        scale_strs = []
        i = idx + 1
        while i < len(args) and not args[i].startswith("--"):
            scale_strs.append(args[i])
            i += 1
        scales = [float(s) for s in scale_strs]
        args = args[:idx] + args[i:]

    paths = [Path(a) for a in args if not a.startswith("--")]
    if not paths:
        print("Usage: python -m scripts.evaluation.sweep_retargeting_scale <session_dirs...>")
        sys.exit(1)

    # Collect session dirs
    sessions = []
    for p in paths:
        if (p / "processed_sequence_with_proxy.npz").exists():
            sessions.append(p)
        else:
            found = sorted(p.rglob("processed_sequence_with_proxy.npz"))
            sessions.extend(f.parent for f in found)

    if not sessions:
        print("No sessions with processed_sequence_with_proxy.npz found.")
        sys.exit(1)

    print(f"\nSweeping scale over {scales}")
    print(f"Sessions: {[s.name for s in sessions]}\n")

    # results[scale] = list of per-session mean errors
    results = {s: [] for s in scales}

    for scale in scales:
        print(f"── scale={scale:.2f} ────────────────────────────────")
        for session_dir in sessions:
            mean_err = _run_one(session_dir, scale)
            results[scale].append(mean_err)
            tag = f"{mean_err:.1f} mm" if np.isfinite(mean_err) else "n/a"
            print(f"  {session_dir.name:<30}  {tag}")

        if not keep:
            for session_dir in sessions:
                f = session_dir / _output_name(scale)
                if f.exists():
                    f.unlink()

    # ── summary table ──────────────────────────────────────────────────────────
    print("\n" + "─" * 52)
    print(f"  {'scale':>6}  " + "  ".join(f"{s.name[:12]:>12}" for s in sessions) + "  overall")
    print("─" * 52)
    for scale in scales:
        vals = results[scale]
        overall = np.nanmean(vals) if any(np.isfinite(v) for v in vals) else np.nan
        row = f"  {scale:>6.2f}  "
        row += "  ".join(f"{v:>12.1f}" if np.isfinite(v) else f"{'n/a':>12}" for v in vals)
        row += f"  {overall:>7.1f}"
        print(row)
    print("─" * 52)

    best_scale = min(scales, key=lambda s: np.nanmean(results[s]))
    print(f"\n  Best scale: {best_scale:.2f}  (overall mean {np.nanmean(results[best_scale]):.1f} mm)\n")


if __name__ == "__main__":
    main()
