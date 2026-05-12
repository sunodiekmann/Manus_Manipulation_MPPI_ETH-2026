"""
Optimize per-finger cube-frame translational offsets that close the systematic
gap between the Manus fingertip landmark and the robot fingertip geom centre.

For statics holds (every fingertip pressed flat against the cube), the robot
fingertip after IK should land on the cube surface. Any signed distance to
the surface is calibration error — landmark sitting inside the pulp, behind
the nail, etc. This script fits a constant per-finger offset (in cube frame,
matching `OFFLINE_TARGET_OFFSETS` semantics in `map_cube_frame_target`) that
drives that error to zero.

Optimizer: scipy.optimize.minimize, Nelder-Mead over 15 variables
            (5 fingers × 3 axes), or 5×3-D in --sequential mode.

The wrist offset is held fixed during optimization — re-run
`optimize_wrist_offset.py` separately if it ever drifts.

Usage:
    python -m scripts.calibration.optimize_finger_offsets
    python -m scripts.calibration.optimize_finger_offsets --sequential
    python -m scripts.calibration.optimize_finger_offsets --frames-per-session 4
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from scripts.retargeting.tesollo_constants import FINGERS, TARGET_SCALE
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
)
from scripts.retargeting.tesollo_runtime_baseline import (
    initialize_targets_from_configuration,
    solve_ik_step,
    update_finger_frame_tasks_from_mocaps,
    update_fingertip_targets_cube_frame,
    update_posture_target_from_joint_angles,
    update_relative_finger_targets,
    update_wrist_task_target,
)
from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.retargeting.retarget_sequence_cube_relative import (
    MODEL_PATH, DEFAULT_WRIST_OFFSET, _apply_wrist_offset,
)
from scripts.utils.config import CUBE_SIZE
from scripts.utils.transforms import pose_to_matrix

STATICS_DIR = Path("recordings/statics")
HALF = CUBE_SIZE / 2

FINGERTIP_GEOMS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]

IK_STEPS = 30
WARMUP_CALLS = 20
IK_DT = 0.002

INITIAL_SIMPLEX_STEP = 0.005   # 5 mm — sensible scale for landmark calibration


# ── geometry ─────────────────────────────────────────────────────────────────

def _signed_dist_box(p: np.ndarray) -> float:
    """True signed distance to axis-aligned cube of half-extent HALF, centred.

    Negative inside, positive outside (true Euclidean magnitude outside).
    """
    q = np.abs(p) - HALF
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(0.0, q.max())
    return float(outside + inside)


# ── data loading ─────────────────────────────────────────────────────────────

def _load_statics_sessions(statics_dir: Path, frames_per_session: int):
    sessions = []
    for sub in sorted(p for p in statics_dir.iterdir() if p.is_dir()):
        proc = sub / "processed_sequence_with_proxy.npz"
        if not proc.exists():
            continue
        d = np.load(proc, allow_pickle=True)
        N = len(d["timeline"])
        idx = np.linspace(0, N - 1, min(frames_per_session, N)).astype(int)
        sessions.append({
            "name": sub.name,
            "fingertips_cube": np.asarray(d["right_fingertips_cube"], dtype=object)[idx],
            "wrist_world":     np.asarray(d["right_wrist_world"],     dtype=object)[idx],
            "ergonomics":      np.asarray(d["right_ergonomics"],      dtype=object)[idx],
            "cube_pose":       np.asarray(d["cube_pose_world"],       dtype=object)[idx],
        })
    return sessions


# ── retargeting (in-memory) ──────────────────────────────────────────────────

def _build_offset_dict(vec: np.ndarray) -> dict:
    """15-vector → {finger_geom_name: np.array(3)}."""
    return {f: vec[3 * i : 3 * i + 3] for i, f in enumerate(FINGERS)}


def _ik_one(model, mj_data, task_bundle, default_quats, fingertips, T_cube,
            T_robot_base, ergonomics, ik_steps, offsets):
    update_fingertip_targets_cube_frame(
        model=model, data=mj_data, task_bundle=task_bundle,
        fingertip_positions_cube=fingertips,
        fingertip_quats=default_quats,
        T_world_from_cube=T_cube,
        T_world_from_wrist=T_robot_base,
        offsets=offsets,
        scale=TARGET_SCALE,
    )
    update_relative_finger_targets(task_bundle)
    update_posture_target_from_joint_angles(task_bundle, ergonomics, T_robot_base)
    update_finger_frame_tasks_from_mocaps(model, mj_data, task_bundle)
    update_wrist_task_target(task_bundle, T_robot_base)
    for _ in range(ik_steps):
        solve_ik_step(task_bundle, dt=IK_DT)
        # Hard-pin wrist translation; orientation stays soft.
        task_bundle.configuration.q[0:3] = T_robot_base[:3, 3]
    mj_data.qpos[: task_bundle.configuration.q.shape[0]] = task_bundle.configuration.q
    mujoco.mj_forward(model, mj_data)


def _retarget_session(model, task_bundle, session, wrist_offset, offsets, ik_steps):
    """Return list of (qpos, T_world_from_cube) tuples for the session."""
    mj_data = task_bundle.configuration.data
    default_quats = np.tile([0., 0., 0., 1.], (len(task_bundle.fingers), 1))

    fingertips_seq = session["fingertips_cube"]
    wrist_seq      = session["wrist_world"]
    ergo_seq       = session["ergonomics"]
    cube_seq       = session["cube_pose"]

    T_cube_0 = pose_to_matrix(np.asarray(cube_seq[0], dtype=float))
    T_robot_0 = _apply_wrist_offset(np.asarray(wrist_seq[0], dtype=float), wrist_offset)
    for _ in range(WARMUP_CALLS):
        _ik_one(model, mj_data, task_bundle, default_quats,
                fingertips_seq[0], T_cube_0, T_robot_0, ergo_seq[0], ik_steps, offsets)

    out = []
    for i in range(len(fingertips_seq)):
        T_cube = pose_to_matrix(np.asarray(cube_seq[i], dtype=float))
        T_robot = _apply_wrist_offset(np.asarray(wrist_seq[i], dtype=float), wrist_offset)
        _ik_one(model, mj_data, task_bundle, default_quats,
                fingertips_seq[i], T_cube, T_robot, ergo_seq[i], ik_steps, offsets)
        out.append((task_bundle.configuration.q.copy(), T_cube.copy()))

    return out


# ── loss ─────────────────────────────────────────────────────────────────────

def _surface_error(model, qpos_cube_pairs, finger_mask: np.ndarray):
    """Mean |signed_dist_to_cube_surface| (metres), gated by `finger_mask`.

    finger_mask : (5,) bool — which fingers contribute to the loss
                  (used by sequential mode to focus on one finger at a time).
    """
    data = mujoco.MjData(model)
    geom_ids = [int(model.geom(name).id) for name in FINGERTIP_GEOMS]

    dists = []
    for qpos, T_cube in qpos_cube_pairs:
        data.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        R_wc = T_cube[:3, :3]
        t_wc = T_cube[:3, 3]
        for f in range(5):
            if not finger_mask[f]:
                continue
            tip_w = np.asarray(data.geom_xpos[geom_ids[f]])
            p_cube = R_wc.T @ (tip_w - t_wc)
            dists.append(abs(_signed_dist_box(p_cube)))

    return float(np.mean(dists)) if dists else float("inf")


# ── main ─────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    fps = 6
    maxiter = 80
    sequential = "--sequential" in argv
    wrist_offset = np.asarray(DEFAULT_WRIST_OFFSET, dtype=float)
    start = np.zeros(15, dtype=float)

    if "--frames-per-session" in argv:
        i = argv.index("--frames-per-session")
        fps = int(argv[i + 1])
    if "--maxiter" in argv:
        i = argv.index("--maxiter")
        maxiter = int(argv[i + 1])
    if "--wrist-offset" in argv:
        i = argv.index("--wrist-offset")
        wrist_offset = np.array(
            [float(argv[i + 1]), float(argv[i + 2]), float(argv[i + 3])]
        )
    return start, fps, maxiter, sequential, wrist_offset


def _print_offsets(label: str, offsets: dict) -> None:
    print(f"\n{label}:")
    for f in FINGERS:
        o = offsets[f] * 1000.0
        print(f"  {f:11s}  ({o[0]:+7.2f}, {o[1]:+7.2f}, {o[2]:+7.2f}) mm")


def _per_finger_residuals(model, qpos_cube_pairs):
    """Per-finger mean signed distance (mm) for diagnostics."""
    data = mujoco.MjData(model)
    geom_ids = [int(model.geom(name).id) for name in FINGERTIP_GEOMS]
    rows = [[] for _ in range(5)]
    for qpos, T_cube in qpos_cube_pairs:
        data.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        R_wc = T_cube[:3, :3]
        t_wc = T_cube[:3, 3]
        for f in range(5):
            tip_w = np.asarray(data.geom_xpos[geom_ids[f]])
            p_cube = R_wc.T @ (tip_w - t_wc)
            rows[f].append(_signed_dist_box(p_cube) * 1000.0)
    return [np.asarray(r) for r in rows]


def main():
    start_flat, frames_per_session, maxiter, sequential, wrist_offset = _parse_args(sys.argv[1:])

    print(f"Wrist offset (held fixed): ({wrist_offset[0]:+.4f}, "
          f"{wrist_offset[1]:+.4f}, {wrist_offset[2]:+.4f}) m")
    print(f"Loading statics sessions (frames/session = {frames_per_session}) …")
    sessions = _load_statics_sessions(STATICS_DIR, frames_per_session)
    if not sessions:
        print(f"No statics sessions found in {STATICS_DIR}")
        sys.exit(1)
    n_total = sum(len(s["fingertips_cube"]) for s in sessions)
    print(f"  {len(sessions)} sessions, {n_total} total frames")

    print("Building task bundle …")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    task_bundle = build_tesollo_task_bundle(model)
    model = task_bundle.configuration.model
    mj_data = task_bundle.configuration.data
    initialize_targets_from_configuration(task_bundle)
    initialize_fingertip_target_mocaps(model, mj_data, task_bundle.fingers)

    eval_count = [0]

    def evaluate(flat_vec, finger_mask):
        offsets = _build_offset_dict(flat_vec)
        pairs = []
        for s in sessions:
            pairs.extend(_retarget_session(model, task_bundle, s, wrist_offset,
                                            offsets, IK_STEPS))
        err = _surface_error(model, pairs, finger_mask)
        eval_count[0] += 1
        return err, pairs

    # ── baseline (zero offsets) ─────────────────────────────────────────────
    print("\nBaseline (zero offsets) …")
    baseline_err, baseline_pairs = evaluate(np.zeros(15), np.ones(5, dtype=bool))
    print(f"  mean |signed dist| = {baseline_err * 1000:.2f} mm")
    per_finger = _per_finger_residuals(model, baseline_pairs)
    for f, name in enumerate(FINGERS):
        r = per_finger[f]
        print(f"  {name:11s}  mean={r.mean():+6.2f} mm  "
              f"|mean|={np.abs(r).mean():5.2f} mm  "
              f"min={r.min():+6.2f}  max={r.max():+6.2f}")

    # ── optimise ────────────────────────────────────────────────────────────
    flat = start_flat.copy()

    if sequential:
        print(f"\nSequential per-finger optimisation (5 × 3-D Nelder-Mead, maxiter={maxiter}) …")
        for f_idx, fname in enumerate(FINGERS):
            mask = np.zeros(5, dtype=bool)
            mask[f_idx] = True
            print(f"\n  → {fname}")

            def loss(x3, _f=f_idx, _m=mask):
                v = flat.copy()
                v[3 * _f : 3 * _f + 3] = x3
                err, _ = evaluate(v, _m)
                print(f"    [{eval_count[0]:4d}]  ({x3[0]:+.4f}, {x3[1]:+.4f}, {x3[2]:+.4f})"
                      f"   err = {err * 1000:.2f} mm")
                return err

            x0 = flat[3 * f_idx : 3 * f_idx + 3]
            result = minimize(
                loss, x0, method="Nelder-Mead",
                options={
                    "xatol": 5e-4, "fatol": 5e-5, "maxiter": maxiter,
                    "initial_simplex": np.vstack([x0, x0 + np.eye(3) * INITIAL_SIMPLEX_STEP]),
                },
            )
            flat[3 * f_idx : 3 * f_idx + 3] = result.x

    else:
        print(f"\nJoint optimisation (15-D Nelder-Mead, maxiter={maxiter}) …")

        def loss(x):
            err, _ = evaluate(x, np.ones(5, dtype=bool))
            o = x * 1000.0
            print(f"  [{eval_count[0]:4d}]  err = {err * 1000:.2f} mm   "
                  f"th=({o[0]:+5.1f},{o[1]:+5.1f},{o[2]:+5.1f})  "
                  f"id=({o[3]:+5.1f},{o[4]:+5.1f},{o[5]:+5.1f})  "
                  f"mi=({o[6]:+5.1f},{o[7]:+5.1f},{o[8]:+5.1f})  "
                  f"ri=({o[9]:+5.1f},{o[10]:+5.1f},{o[11]:+5.1f})  "
                  f"li=({o[12]:+5.1f},{o[13]:+5.1f},{o[14]:+5.1f}) mm")
            return err

        simplex = np.vstack([flat, flat + np.eye(15) * INITIAL_SIMPLEX_STEP])
        result = minimize(
            loss, flat, method="Nelder-Mead",
            options={"xatol": 5e-4, "fatol": 5e-5, "maxiter": maxiter,
                     "initial_simplex": simplex},
        )
        flat = result.x

    # ── final ───────────────────────────────────────────────────────────────
    final_offsets = _build_offset_dict(flat)
    _print_offsets("Optimal per-finger offsets (cube frame, mm)", final_offsets)

    print("\nFinal evaluation …")
    final_err, final_pairs = evaluate(flat, np.ones(5, dtype=bool))
    print(f"  mean |signed dist| = {final_err * 1000:.2f} mm  "
          f"(baseline {baseline_err * 1000:.2f} mm)")
    per_finger = _per_finger_residuals(model, final_pairs)
    for f, name in enumerate(FINGERS):
        r = per_finger[f]
        print(f"  {name:11s}  mean={r.mean():+6.2f} mm  "
              f"|mean|={np.abs(r).mean():5.2f} mm  "
              f"min={r.min():+6.2f}  max={r.max():+6.2f}")

    print(f"\nTotal evaluations: {eval_count[0]}")

    # Snippet to paste into retarget_sequence_cube_relative.py.
    print("\nPaste into scripts/retargeting/retarget_sequence_cube_relative.py:\n")
    print("OFFLINE_TARGET_OFFSETS = {")
    for f in FINGERS:
        o = final_offsets[f]
        print(f"    \"{f}\": np.array([{o[0]:+.5f}, {o[1]:+.5f}, {o[2]:+.5f}]),")
    print("}")


if __name__ == "__main__":
    main()
