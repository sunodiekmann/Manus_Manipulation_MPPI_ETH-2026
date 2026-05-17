"""
Optimize the wrist→robot-base translational offset (in wrist frame) by
minimizing the robot-fingertip calibration error on the static-pose recordings.

The Manus wrist landmark and the robot freejoint origin do not coincide
(Manus tracks somewhere on the wrist/palm; the robot freejoint is at the
bottom of the mount cylinder). Setting the freejoint = Manus wrist directly
introduces a systematic offset in retargeted fingertip positions. This
script finds the (tx, ty, tz) translation that, when applied in the wrist
frame, makes the robot fingertips land closest to the cube surface in the
statics holds.

For each candidate offset:
  1. Retarget every statics session with that offset (in memory, no disk I/O)
  2. Compute robot fingertip positions in cube frame via FK
  3. Mean |signed-distance| from each fingertip to nearest cube face → loss

Optimizer: scipy.optimize.minimize (Nelder-Mead).

Usage:
    python -m scripts.calibration.optimize_wrist_offset
    python -m scripts.calibration.optimize_wrist_offset --start 0 0 -0.05
    python -m scripts.calibration.optimize_wrist_offset --frames-per-session 4
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from scripts.retargeting.tesollo_constants import TARGET_SCALE
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_hand_base_pose,
)
from scripts.retargeting.tesollo_ik_runtime import (
    initialize_targets_from_configuration,
    solve_ik_step,
    update_finger_frame_tasks_from_mocaps,
    update_fingertip_targets_cube_frame,
    update_posture_target_from_joint_angles,
    update_relative_finger_targets,
    update_wrist_task_target,
)
from scripts.retargeting.tesollo_calibration import (
    OFFLINE_TARGET_OFFSETS, apply_wrist_offset,
)
from scripts.retargeting.tesollo_mink_ik import (
    MODEL_PATH, _pin_wrist_in_configuration,
)
from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.utils.config import CUBE_SIZE
from scripts.utils.transforms import pose_to_matrix

STATICS_DIR = Path("recordings/statics")
HALF = CUBE_SIZE / 2

FINGERTIP_GEOMS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]

IK_STEPS = 30          # per frame during optimization (warm-start helps)
WARMUP_CALLS = 20      # warm-up calls when entering a session
IK_DT = 0.002


# ── geometry ─────────────────────────────────────────────────────────────────

def _signed_dist_to_surface(p: np.ndarray) -> float:
    face_signed = np.array([
        p[0] - HALF, -p[0] - HALF,
        p[1] - HALF, -p[1] - HALF,
        p[2] - HALF, -p[2] - HALF,
    ])
    idx = int(np.argmin(np.abs(face_signed)))
    return float(face_signed[idx])


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

def _ik_one(model, mj_data, task_bundle, default_quats, fingertips, T_cube,
            T_robot_base, ergonomics, ik_steps):
    update_fingertip_targets_cube_frame(
        model=model, data=mj_data, task_bundle=task_bundle,
        fingertip_positions_cube=fingertips,
        fingertip_quats=default_quats,
        T_world_from_cube=T_cube,
        T_world_from_wrist=T_robot_base,
        offsets=OFFLINE_TARGET_OFFSETS,
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


def _retarget_session(model, task_bundle, session, wrist_offset, ik_steps):
    """Retarget all loaded frames of one session in memory.

    Returns list of (qpos, T_world_from_cube) tuples, one per frame.
    """
    mj_data = task_bundle.configuration.data
    default_quats = np.tile([0., 0., 0., 1.], (len(task_bundle.fingers), 1))

    fingertips_seq = session["fingertips_cube"]
    wrist_seq      = session["wrist_world"]
    ergo_seq       = session["ergonomics"]
    cube_seq       = session["cube_pose"]

    # Warm-up on the first frame so IK converges from whatever previous state.
    T_cube_0 = pose_to_matrix(np.asarray(cube_seq[0], dtype=float))
    T_robot_0 = apply_wrist_offset(np.asarray(wrist_seq[0], dtype=float), wrist_offset)
    for _ in range(WARMUP_CALLS):
        _ik_one(model, mj_data, task_bundle, default_quats,
                fingertips_seq[0], T_cube_0, T_robot_0, ergo_seq[0], ik_steps)

    out = []
    for i in range(len(fingertips_seq)):
        T_cube = pose_to_matrix(np.asarray(cube_seq[i], dtype=float))
        T_robot = apply_wrist_offset(np.asarray(wrist_seq[i], dtype=float), wrist_offset)
        _ik_one(model, mj_data, task_bundle, default_quats,
                fingertips_seq[i], T_cube, T_robot, ergo_seq[i], ik_steps)
        out.append((task_bundle.configuration.q.copy(), T_cube.copy()))

    return out


# ── calibration error ────────────────────────────────────────────────────────

def _precompute_targets(sessions):
    """For each session: target world positions per frame, shape (Nf, 5, 3)."""
    out = []
    for s in sessions:
        Nf = len(s["fingertips_cube"])
        targets = np.zeros((Nf, 5, 3))
        for i in range(Nf):
            T_cube = pose_to_matrix(np.asarray(s["cube_pose"][i], dtype=float))
            R_wc = T_cube[:3, :3]
            t_wc = T_cube[:3, 3]
            for f in range(5):
                p_cube = np.asarray(s["fingertips_cube"][i][f], dtype=float) * TARGET_SCALE
                targets[i, f] = R_wc @ p_cube + t_wc
        out.append(targets)
    return out


def _calibration_error(model, qpos_cube_pairs, target_world_seq):
    """Mean Euclidean distance (m) from each robot fingertip to its IK target.

    Targets are the human fingertip positions expressed in world frame
    (T_world_cube @ pos_cube * scale). This directly measures how well the robot
    reproduces the human fingertip pattern — the right thing to drive the offset.
    """
    data = mujoco.MjData(model)
    geom_ids = [int(model.geom(name).id) for name in FINGERTIP_GEOMS]

    dists = []
    targets_flat = np.concatenate(target_world_seq, axis=0)   # (sum Nf, 5, 3)
    for k, (qpos, _T_cube) in enumerate(qpos_cube_pairs):
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        tip_world = np.array([data.geom_xpos[g] for g in geom_ids])  # (5, 3)
        d = np.linalg.norm(tip_world - targets_flat[k], axis=1)      # (5,)
        dists.extend(d.tolist())

    return float(np.mean(dists)) if dists else float("inf")


# ── main ─────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    start = np.zeros(3, dtype=float)
    fps = 6
    maxiter = 60

    if "--start" in argv:
        i = argv.index("--start")
        start = np.array([float(argv[i+1]), float(argv[i+2]), float(argv[i+3])])
    if "--frames-per-session" in argv:
        i = argv.index("--frames-per-session")
        fps = int(argv[i+1])
    if "--maxiter" in argv:
        i = argv.index("--maxiter")
        maxiter = int(argv[i+1])
    return start, fps, maxiter


def main():
    start_offset, frames_per_session, maxiter = _parse_args(sys.argv[1:])

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

    print("Precomputing target world positions …")
    targets_per_session = _precompute_targets(sessions)

    eval_count = [0]

    def loss(offset):
        pairs = []
        for s in sessions:
            pairs.extend(_retarget_session(model, task_bundle, s, offset, IK_STEPS))
        err = _calibration_error(model, pairs, targets_per_session)
        eval_count[0] += 1
        print(f"  [{eval_count[0]:3d}]  offset=({offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f})"
              f"   mean target-gap = {err*1000:.2f} mm")
        return err

    print(f"\nStarting Nelder-Mead from offset={start_offset.tolist()}\n")
    result = minimize(
        loss, start_offset, method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": maxiter,
                 "initial_simplex": np.vstack([start_offset, start_offset + np.eye(3)*0.03])},
    )

    print(f"\nOptimal offset: ({result.x[0]:+.4f}, {result.x[1]:+.4f}, {result.x[2]:+.4f}) m")
    print(f"Final mean |err|: {result.fun*1000:.2f} mm  (after {eval_count[0]} evals)")
    print(f"\nUse with:")
    print(f"  --offset {result.x[0]:.4f} {result.x[1]:.4f} {result.x[2]:.4f}")


if __name__ == "__main__":
    main()
