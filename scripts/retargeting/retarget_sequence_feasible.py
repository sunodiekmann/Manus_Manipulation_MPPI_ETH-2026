"""Stage 2c — Feasibility-aware retargeting (Pass 1 cube smoothing + Pass 2 IK).

Builds on Stage 2a (`retarget_sequence_cube_relative`) and Stage 2b
(`spider_contact`). Produces a per-frame qpos trajectory with smooth
snap-to-surface for intended contacts plus a lightly nudged cube pose
trajectory that absorbs residual robot-vs-human geometry mismatch.

Pipeline:
    Pass 1: scripts.retargeting.feasible_retargeting.cube_smoother
        Sequence-wide L-BFGS-B over per-frame (δ_pos, δ_rot_aa) bounded by
        ±5 mm / ±5°, minimising in-contact target↔robot-tip distance with
        a smoothness regulariser. Outputs `cube_pose_smoothed (N, 7)`.
    Pass 2: per-frame mink IK using:
        - smoothed cube pose for target placement
        - `CollisionAvoidanceLimit` for fingertips (light) and non-tips (hard)
        - smooth snap-to-surface for sustained contact + active pushes
        - warm start from previous frame's solution

Inputs (per session):
    recordings/<session>/retargeted_sequence_cube_relative.npz
    recordings/<session>/contact_sequence.npz
    recordings/<session>/processed_sequence_with_proxy.npz  (raw fingertip cube coords)

Output:
    recordings/<session>/retargeted_sequence_feasible.npz
        timeline          (N,)
        retargeted_qpos   (N, 27)        — Stage 2c IK solution
        cube_pose_smoothed (N, 7)        — Pass 1 output
        snap_weight       (N, 5) float   — smooth snap blend weight per finger
        wrist_offset      (3,)           — calibration constant used

Usage:
    python -m scripts.retargeting.retarget_sequence_feasible recordings/<session>
    python -m scripts.retargeting.retarget_sequence_feasible recordings/<session> --force
    python -m scripts.retargeting.retarget_sequence_feasible recordings/<session> --no-smooth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.feasible_retargeting.cube_smoother import smooth_cube_pose
from scripts.retargeting.retarget_sequence_cube_relative import (
    DEFAULT_WRIST_OFFSET,
    MODEL_PATH,
    OFFLINE_TARGET_OFFSETS,
    _apply_wrist_offset,
)
from scripts.retargeting.tesollo_constants import (
    FEASIBILITY_NONTIP_PAIRS,
    FEASIBILITY_TIP_PAIRS,
    FINGERS,
    TARGET_SCALE,
)
from scripts.retargeting.tesollo_ik_runtime import (
    initialize_targets_from_configuration,
    reset_relative_finger_targets,
    solve_ik_step,
    update_finger_frame_tasks_from_mocaps,
    update_posture_target_from_joint_angles,
    update_relative_finger_targets,
    update_wrist_task_target,
)
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_mocap_body_pose,
    set_mocap_body_pose_from_matrix,
)
from scripts.retargeting.tesollo_target_mapping import (
    _CUBE_HALF,
    map_cube_frame_target,
    rotate_local_fingertip_quaternion,
)
from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.utils.transforms import pose_to_matrix

IK_STEPS = 50
IK_DT = 0.002
WARMUP_CALLS = 30

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


# ── helpers ──────────────────────────────────────────────────────────────────


def _matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """4×4 → [x,y,z,qw,qx,qy,qz]."""
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return np.array([pos[0], pos[1], pos[2],
                     quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _build_intent_targets(
    fingertips_cube_seq,     # (N,) of (5, 3) or None
    cube_pose_seq,            # (N, 7)
    wrist_pose_seq,           # (N,) of (4, 4) or pose vector or None
    scale: float = TARGET_SCALE,
    offsets: dict = OFFLINE_TARGET_OFFSETS,
    sd_threshold_m: float = 0.020,
) -> np.ndarray:
    """Per-frame, per-finger "intended contact target" in cube frame.

    Builds the same cube-frame target as map_cube_frame_target would (raw
    fingertip × scale + wrist-frame offset rotated wrist→cube), then masks
    to NaN where the signed distance to the nearest cube face exceeds
    `sd_threshold_m`. Used as the Pass 1 smoother's contact-intent signal —
    cube nudges to align the robot tip with this target when within band.
    """
    N = cube_pose_seq.shape[0]
    out = np.full((N, len(FINGERS), 3), np.nan)
    for t in range(N):
        cp = cube_pose_seq[t]
        ft = fingertips_cube_seq[t]
        wp = wrist_pose_seq[t]
        if np.any(np.isnan(cp)) or ft is None or wp is None:
            continue
        T_wc = pose_to_matrix(cp)
        T_ww = _wrist_pose_to_matrix(wp)
        if T_ww is None:
            continue
        R_world_cube = T_wc[:3, :3]
        R_world_wrist = T_ww[:3, :3]
        ft_arr = np.asarray(ft, dtype=float)
        if np.any(np.isnan(ft_arr)):
            continue
        for f, finger in enumerate(FINGERS):
            offset_cube = R_world_cube.T @ (R_world_wrist @ offsets[finger])
            p = ft_arr[f] * scale + offset_cube
            sd = float(np.abs(p).max() - _CUBE_HALF)
            if sd <= sd_threshold_m:
                out[t, f] = p
    return out


def _smooth_binary_mask(mask: np.ndarray, transition_frames: int) -> np.ndarray:
    """Triangular dilation of a (N, F) bool mask → (N, F) weight in [0, 1].

    Each True frame stays at weight 1; weight ramps linearly to 0 over
    `transition_frames` frames on each side. Overlapping ramps take the max,
    so a long True region blends smoothly with its neighbours' ramps.
    """
    if transition_frames <= 0:
        return mask.astype(float)
    N = mask.shape[0]
    out = mask.astype(float)
    src = mask.astype(float)
    for dt in range(1, transition_frames + 1):
        ramp = 1.0 - dt / (transition_frames + 1)
        forward = np.zeros_like(out)
        forward[dt:] = src[:-dt] * ramp      # True at τ → weight at τ+dt
        out = np.maximum(out, forward)
        backward = np.zeros_like(out)
        backward[:N - dt] = src[dt:] * ramp  # True at τ → weight at τ-dt
        out = np.maximum(out, backward)
    return out


def _compute_targets(
    fingertips_cube_seq,                # (N,) of (5, 3) or None
    T_world_cube_seq,                    # (N,) of (4,4) or None
    T_world_wrist_seq,                   # (N,) of (4,4) or None
    timeline: np.ndarray,                 # (N,)
    scale: float,
    offsets: dict,
    snap_band_m: float,
    min_drop_m: float,
    surface_band_m: float,
    snap_transition_frames: int,
    approach_window_frames: int = 5,
    manual_constraints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-frame, per-finger IK target sequence in cube frame.

    Pipeline per (t, f):
        1. Apply scale + wrist-frame offset → raw cube-frame target.
        2. Existing inside-cube clamp.
        3. Snapped variant: project onto the nearest face exactly.
        4. Cumulative drop = SD[t-K] − SD[t] over `approach_window_frames` —
           how much closer the finger has gotten over the lookback window.
           Robust to mocap jitter (hover oscillates ±2 mm; a push drops 5+ mm).
        5. Binary intent — snap if EITHER:
              (a) sd ≤ surface_band      — sustained contact at the surface,
              (b) sd ≤ snap_band AND cum_drop > min_drop — active push.
        6. Temporal smoothing of binary → continuous weight ∈ [0, 1].
        7. Interpolate raw vs snapped by the smoothed weight.

    The interpolated target eliminates the "snap on/off" jolt that would
    otherwise show up at the start/end of each push.

    Returns
    -------
    final_target_cube : (N, 5, 3) — IK target in cube frame, NaN-bearing.
    sd                : (N, 5) — signed distance of the raw target to surface
                                  (diagnostic; negative=inside, positive=outside).
    approach          : (N, 5) — −d(SD)/dt in m/s; positive = closing.
    snap_weight       : (N, 5) — smoothed weight applied to the snap.
    """
    N = timeline.shape[0]
    raw_target_cube = np.full((N, 5, 3), np.nan)
    snapped_target_cube = np.full((N, 5, 3), np.nan)
    sd = np.full((N, 5), np.nan)

    from scripts.retargeting.tesollo_target_mapping import _clamp_target_outside_cube

    for t in range(N):
        T_wc = T_world_cube_seq[t]
        T_ww = T_world_wrist_seq[t]
        ft = fingertips_cube_seq[t]
        if T_wc is None or T_ww is None or ft is None:
            continue
        R_world_cube = T_wc[:3, :3]
        R_world_wrist = T_ww[:3, :3]
        ft_arr = np.asarray(ft, dtype=float)
        if np.any(np.isnan(ft_arr)):
            continue
        for f, finger in enumerate(FINGERS):
            offset_cube = R_world_cube.T @ (R_world_wrist @ offsets[finger])
            p = ft_arr[f] * scale + offset_cube
            p_clamped = _clamp_target_outside_cube(p)
            raw_target_cube[t, f] = p_clamped
            # Snapped variant: project onto a face (SD = 0). A manual
            # face-bound constraint overrides argmax so the IK pulls the tip
            # onto the constrained face exclusively.
            abs_p = np.abs(p_clamped)
            forced_face = (
                int(manual_constraints[t, f])
                if manual_constraints is not None
                and manual_constraints[t, f] >= 0
                else -1
            )
            if forced_face >= 0:
                from scripts.retargeting.manual_contacts import project_target_onto_face
                snapped = project_target_onto_face(p_clamped, forced_face, _CUBE_HALF)
            else:
                idx = int(np.argmax(abs_p))
                sign = 1.0 if p_clamped[idx] >= 0.0 else -1.0
                snapped = p_clamped.copy()
                snapped[idx] = sign * _CUBE_HALF
            snapped_target_cube[t, f] = snapped
            sd[t, f] = float(abs_p.max() - _CUBE_HALF)

    # Cumulative drop = SD[t-K] - SD[t]; positive means the finger has gotten
    # closer by that much (metres) over the past K frames. Way more robust
    # than instantaneous −d(SD)/dt because mocap jitter averages out over the
    # window, while a real push leaves a clear net decrease.
    K = max(1, int(approach_window_frames))
    cum_drop = np.zeros_like(sd)
    for t in range(K, N):
        prev = sd[t - K]
        cur = sd[t]
        if np.any(np.isnan(prev)) or np.any(np.isnan(cur)):
            continue
        cum_drop[t] = prev - cur
    # Backfill the first K frames with the K-th frame's value so they aren't
    # an inert dead zone — but if the first K frames are stable, drop will
    # be near zero anyway.
    if N > K:
        cum_drop[:K] = cum_drop[K]

    # Diagnostic: also report the instantaneous derivative (not used in gate).
    approach = np.zeros_like(sd)
    dt = np.diff(timeline)
    for t in range(1, N):
        if np.isnan(sd[t]).all() or np.isnan(sd[t - 1]).all():
            continue
        step = dt[t - 1] if dt[t - 1] > 1e-6 else 1.0 / 30.0
        approach[t] = (sd[t - 1] - sd[t]) / step
    if N >= 2:
        approach[0] = approach[1]

    # Binary intent → smoothed weight. Two disjoint cases:
    #   (a) sustained contact: target sits within `surface_band_m` of the
    #       cube surface — assumed firmly intended (e.g. thumb pressed in).
    #   (b) active push: target is within `snap_band_m` and has dropped at
    #       least `min_drop_m` over the lookback window — discriminator
    #       against hover jitter that lacks net displacement.
    if snap_band_m > 0.0:
        sustained = (sd > 0.0) & (sd <= surface_band_m)
        active_push = (sd > 0.0) & (sd <= snap_band_m) & (cum_drop > min_drop_m)
        binary = sustained | active_push
        binary = np.where(np.isnan(sd), False, binary)
        # Forbidden (manual_constraint == -2): no snap, no contact.
        if manual_constraints is not None:
            from scripts.retargeting.manual_contacts import FORBIDDEN
            binary = binary & (manual_constraints != FORBIDDEN)
        snap_weight = _smooth_binary_mask(binary, snap_transition_frames)
    else:
        snap_weight = np.zeros_like(sd)

    # Interpolate: w=0 → recorded, w=1 → fully snapped to surface.
    final_target_cube = raw_target_cube.copy()
    for t in range(N):
        for f in range(5):
            w = snap_weight[t, f]
            if w <= 0.0:
                continue
            raw = raw_target_cube[t, f]
            snap = snapped_target_cube[t, f]
            if np.any(np.isnan(raw)) or np.any(np.isnan(snap)):
                continue
            final_target_cube[t, f] = (1.0 - w) * raw + w * snap
    return final_target_cube, sd, approach, snap_weight


def _wrist_pose_to_matrix(wrist_pose) -> np.ndarray | None:
    """Pose vector or 4×4 → 4×4 homogeneous transform."""
    if wrist_pose is None:
        return None
    arr = np.asarray(wrist_pose, dtype=float)
    if arr.shape == (4, 4):
        return arr
    return pose_to_matrix(arr)


def _set_cube_target(
    model, data, T_world_cube, T_world_wrist, finger: str,
    pos_cube: np.ndarray, finger_quat: np.ndarray,
):
    """Place a finger's IK mocap target from a pre-computed cube-frame position.

    Used uniformly for both interpolated recorded targets (Stage 2c snap-aware
    output of `_compute_targets`) and face-transition lift targets. The
    orientation is wrist-derived so the orientation cost lands in the same
    local minimum as the rest of the pipeline.
    """
    R_wc = T_world_cube[:3, :3]
    t_wc = T_world_cube[:3, 3]
    pos_world = R_wc @ np.asarray(pos_cube, dtype=float) + t_wc

    R_ww = T_world_wrist[:3, :3]
    q_local = rotate_local_fingertip_quaternion(finger_quat, finger)
    q_world = R.from_matrix(R_ww) * q_local
    set_mocap_body_pose(model, data, f"{finger}_target", pos_world, q_world.as_quat())


def _feasible_ik_frame(
    model, mj_data, task_bundle,
    target_cube_for_frame: np.ndarray,        # (5, 3) — pre-blended cube-frame targets
    T_world_cube: np.ndarray,                   # (4, 4)
    T_world_wrist: np.ndarray,                  # (4, 4)
    ergonomics: np.ndarray | None,
    finger_quats: np.ndarray,                   # (5, 4) xyzw
    wrist_offset: np.ndarray,
    ik_steps: int = IK_STEPS,
) -> np.ndarray:
    """One frame of Stage 2c IK. The target array already contains:
        - the recorded fingertip target (Stage 2a logic),
        - smooth snap-to-surface when intended contact is detected,
        - smooth face-transition lift trajectories,
    all blended together upstream so the IK only sees a continuous target.
    """
    T_world_robot_base = _apply_wrist_offset(T_world_wrist, wrist_offset)
    set_mocap_body_pose_from_matrix(model, mj_data, "tracked_cube", T_world_cube)

    has_any_target = False
    for f, finger in enumerate(task_bundle.fingers):
        pos_cube = target_cube_for_frame[f]
        if np.any(np.isnan(pos_cube)):
            continue
        _set_cube_target(
            model, mj_data, T_world_cube, T_world_wrist, finger,
            pos_cube, finger_quats[f],
        )
        has_any_target = True

    if has_any_target:
        update_relative_finger_targets(task_bundle)
    else:
        reset_relative_finger_targets(task_bundle)

    update_posture_target_from_joint_angles(task_bundle, ergonomics, T_world_robot_base)
    update_finger_frame_tasks_from_mocaps(model, mj_data, task_bundle)
    update_wrist_task_target(task_bundle, T_world_robot_base)

    for _ in range(ik_steps):
        solve_ik_step(task_bundle, dt=IK_DT)

    mj_data.qpos[: task_bundle.configuration.q.shape[0]] = task_bundle.configuration.q
    mujoco.mj_forward(model, mj_data)
    return task_bundle.configuration.q.copy()


# ── orchestration ────────────────────────────────────────────────────────────


def run_session(
    session_dir: Path,
    force: bool = False,
    do_smooth: bool = True,
    posture_cost: float = 1.0,
    snap_band_m: float = 0.020,
    min_drop_m: float = 0.003,
    surface_band_m: float = 0.003,
    snap_transition_frames: int = 5,
    approach_window_frames: int = 5,
    nontip_min_distance: float = 0.005,
    nontip_detection_distance: float = 0.015,
    tip_min_distance: float = -0.001,
    tip_detection_distance: float = 0.005,
    wrist_offset: np.ndarray = DEFAULT_WRIST_OFFSET,
):
    cube_rel_path = session_dir / "retargeted_sequence_cube_relative.npz"
    proc_path = session_dir / "processed_sequence_with_proxy.npz"
    out_path = session_dir / "retargeted_sequence_feasible.npz"

    for p, label in [(cube_rel_path, "Stage 2a"), (proc_path, "Stage 1")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} input: {p}")
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (--force to overwrite)")
        return

    rel = np.load(cube_rel_path)
    timeline = np.asarray(rel["timeline"], dtype=float)
    qpos_rel = np.asarray(rel["retargeted_qpos"], dtype=float)             # (N, 27)
    cube_pose_recorded = np.asarray(rel["cube_pose_used"], dtype=float)    # (N, 7)
    N = len(timeline)

    proc = np.load(proc_path, allow_pickle=True)
    right_fingertips_cube = proc["right_fingertips_cube"]
    right_wrist_world = proc["right_wrist_world"]
    right_ergonomics = proc["right_ergonomics"]

    # Optional manual contact constraints sidecar.
    from scripts.retargeting.manual_contacts import load_constraints, FORBIDDEN
    manual_constraints = load_constraints(session_dir, timeline)
    if manual_constraints is not None:
        n_forbidden = int((manual_constraints == FORBIDDEN).sum())
        n_facebound = int((manual_constraints >= 0).sum())
        print(f"  manual contact constraints: {n_forbidden} forbid-frames, "
              f"{n_facebound} face-bound frames")

    # ── Pass 1: cube pose smoothing ────────────────────────────────────────
    # Stage 2c is self-contained — derives "intended contact" from SD on the
    # per-frame raw target (offset+scale-adjusted recorded fingertip in cube
    # frame). No dependency on spider_contact, so spider_contact can be run
    # afterwards on the Stage 2c output instead of feeding Stage 2c.
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    if do_smooth:
        t0 = time.perf_counter()
        intent_targets = _build_intent_targets(
            right_fingertips_cube, cube_pose_recorded, right_wrist_world,
            sd_threshold_m=snap_band_m,
        )
        cube_pose_smoothed, smooth_info = smooth_cube_pose(
            model, qpos_rel, cube_pose_recorded, intent_targets,
            verbose=True,
        )
        print(f"  Pass 1 done in {time.perf_counter() - t0:.1f}s")
    else:
        print("  [--no-smooth] skipping Pass 1; using recorded cube poses")
        cube_pose_smoothed = cube_pose_recorded.copy()
        smooth_info = {"skipped": True}

    # ── Pass 2: per-frame IK with split collision avoidance ────────────────
    # Build the bundle without auto-collision so we can stack two limits with
    # different min-distances: hard clearance for non-tips, near-zero for tips.
    # Higher posture cost than Stage 2a (default 1e-3) so the IK strongly
    # prefers the human ergonomics posture and doesn't elbow-flip between
    # frames when the fingertip task has multiple valid joint configurations.
    task_bundle = build_tesollo_task_bundle(
        model, use_collision_avoidance=False, posture_cost=posture_cost,
    )
    model = task_bundle.configuration.model
    mj_data = task_bundle.configuration.data
    task_bundle.limits.append(mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=FEASIBILITY_NONTIP_PAIRS,
        minimum_distance_from_collisions=nontip_min_distance,
        collision_detection_distance=nontip_detection_distance,
    ))
    task_bundle.limits.append(mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=FEASIBILITY_TIP_PAIRS,
        minimum_distance_from_collisions=tip_min_distance,
        collision_detection_distance=tip_detection_distance,
    ))
    print(f"  collision limits: non-tip min={nontip_min_distance*1000:+.1f} mm "
          f"detect={nontip_detection_distance*1000:.1f} mm  |  "
          f"tip min={tip_min_distance*1000:+.1f} mm "
          f"detect={tip_detection_distance*1000:.1f} mm")
    initialize_targets_from_configuration(task_bundle)
    initialize_fingertip_target_mocaps(model, mj_data, task_bundle.fingers)

    finger_quats = np.tile(_IDENTITY_QUAT_XYZW, (len(task_bundle.fingers), 1))

    # Precompute T_world_cube matrices and T_world_wrist matrices.
    T_world_cube_arr = np.empty(N, dtype=object)
    T_world_wrist_arr = np.empty(N, dtype=object)
    for t in range(N):
        cp = cube_pose_smoothed[t]
        if not np.any(np.isnan(cp)):
            T_world_cube_arr[t] = pose_to_matrix(cp)
        T_world_wrist_arr[t] = _wrist_pose_to_matrix(right_wrist_world[t])

    # Precompute per-frame cube-frame IK targets, with smooth snap-to-surface
    # blending (binary intent → triangular dilation → linear interpolation).
    final_target_cube, sd, approach, snap_weight = _compute_targets(
        right_fingertips_cube, T_world_cube_arr, T_world_wrist_arr,
        timeline,
        scale=TARGET_SCALE,
        offsets=OFFLINE_TARGET_OFFSETS,
        snap_band_m=snap_band_m,
        min_drop_m=min_drop_m,
        surface_band_m=surface_band_m,
        snap_transition_frames=snap_transition_frames,
        approach_window_frames=approach_window_frames,
        manual_constraints=manual_constraints,
    )
    if snap_band_m > 0.0:
        n_partial = int((snap_weight > 0).sum())
        n_full = int((snap_weight >= 0.999).sum())
        per_finger_partial = (snap_weight > 0).sum(axis=0)
        names = [f.split("_")[0] for f in FINGERS]
        print(f"  snap-to-surface: {n_partial} (frame,finger) with w>0 "
              f"(of which {n_full} fully snapped)")
        print(f"    surface_band={surface_band_m*1000:.1f}mm | "
              f"snap_band={snap_band_m*1000:.1f}mm "
              f"+ min_drop={min_drop_m*1000:.1f}mm/{approach_window_frames} frames | "
              f"transition=±{snap_transition_frames} frames")
        print(f"  snap per finger {dict(zip(names, per_finger_partial.tolist()))}")
    else:
        print("  snap-to-surface: disabled (--snap-band 0)")

    # Warm up on the first frame that has cube + wrist tracked.
    first_i = next(
        (i for i in range(N)
         if T_world_cube_arr[i] is not None and T_world_wrist_arr[i] is not None
         and right_fingertips_cube[i] is not None),
        None,
    )
    if first_i is None:
        raise RuntimeError("no frame with cube + wrist + fingertips all tracked")

    print(f"  Warming up Pass 2 on frame {first_i} ({WARMUP_CALLS} × {IK_STEPS} IK steps)")
    for _ in range(WARMUP_CALLS):
        _feasible_ik_frame(
            model, mj_data, task_bundle,
            target_cube_for_frame=final_target_cube[first_i],
            T_world_cube=T_world_cube_arr[first_i],
            T_world_wrist=T_world_wrist_arr[first_i],
            ergonomics=right_ergonomics[first_i],
            finger_quats=finger_quats,
            wrist_offset=wrist_offset,
        )

    print(f"  Pass 2: retargeting {N} frames (IK_STEPS={IK_STEPS}) …")
    qpos_out = np.full((N, model.nq), np.nan)

    t0 = time.perf_counter()
    for t in range(N):
        if (T_world_cube_arr[t] is None or T_world_wrist_arr[t] is None
                or right_fingertips_cube[t] is None):
            continue
        qpos_out[t] = _feasible_ik_frame(
            model, mj_data, task_bundle,
            target_cube_for_frame=final_target_cube[t],
            T_world_cube=T_world_cube_arr[t],
            T_world_wrist=T_world_wrist_arr[t],
            ergonomics=right_ergonomics[t],
            finger_quats=finger_quats,
            wrist_offset=wrist_offset,
        )
        if (t + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (t + 1) / elapsed
            eta = (N - t - 1) / rate if rate > 0 else 0.0
            print(f"    [{t + 1:4d}/{N}]  rate={rate:.1f} fr/s  ETA={eta:.0f}s")
    print(f"  Pass 2 done in {time.perf_counter() - t0:.1f}s "
          f"({N / max(time.perf_counter() - t0, 1e-6):.1f} fr/s)")

    np.savez_compressed(
        out_path,
        timeline=timeline,
        retargeted_qpos=qpos_out[:, :27],
        cube_pose_smoothed=cube_pose_smoothed,
        snap_weight=snap_weight,
        wrist_offset=np.asarray(wrist_offset),
        snap_band_m=np.asarray(snap_band_m),
        min_drop_m=np.asarray(min_drop_m),
        surface_band_m=np.asarray(surface_band_m),
        snap_transition_frames=np.asarray(snap_transition_frames),
        approach_window_frames=np.asarray(approach_window_frames),
        nontip_min_distance=np.asarray(nontip_min_distance),
        tip_min_distance=np.asarray(tip_min_distance),
    )
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip Pass 1 cube smoothing; use recorded cube poses")
    parser.add_argument("--posture-cost", type=float, default=1.0,
                        help="Cost weight on the posture (ergonomics) task. "
                             "Higher = IK prefers staying near the human pose, "
                             "suppressing elbow flips during regrabs. Default "
                             "1.0 (Stage 2a uses 1e-3).")
    parser.add_argument("--snap-band", type=float, default=0.020,
                        help="Outside band (m) within which a closing target is "
                             "snapped onto the cube surface. 0 disables snap.")
    parser.add_argument("--min-drop", type=float, default=0.003,
                        help="Minimum net SD drop (m) required over the "
                             "lookback window for a snap to engage. Default "
                             "3 mm — distinguishes real pushes (5-10 mm "
                             "approach) from hover jitter (±2 mm).")
    parser.add_argument("--approach-window", type=int, default=5,
                        help="Frames over which the cumulative drop is "
                             "measured. Larger = more robust to noise, "
                             "smaller = catches shorter pushes.")
    parser.add_argument("--surface-band", type=float, default=0.003,
                        help="A target within this distance (m) of the cube "
                             "surface always snaps (sustained-contact case). "
                             "Catches the always-pressed case the cumulative-"
                             "drop gate misses because there's no motion.")
    parser.add_argument("--snap-transition-frames", type=int, default=5,
                        help="Frames each side over which the snap weight "
                             "ramps 0↔1, eliminating snap-toggle jolts in the "
                             "IK target trajectory. Set 0 for hard binary.")
    parser.add_argument("--nontip-min", type=float, default=0.005,
                        help="Hard clearance (m) between non-tip finger/palm "
                             "geoms and the cube. Default 5 mm.")
    parser.add_argument("--nontip-detect", type=float, default=0.015,
                        help="Distance at which non-tip avoidance activates (m).")
    parser.add_argument("--tip-min", type=float, default=-0.001,
                        help="Min SD between fingertip geoms and the cube. "
                             "Negative allows light contact. Default −1 mm.")
    parser.add_argument("--tip-detect", type=float, default=0.005,
                        help="Distance at which tip avoidance activates (m).")
    args = parser.parse_args()

    for d in args.session_dirs:
        print(f"\n{d.name}")
        run_session(
            d,
            force=args.force,
            do_smooth=not args.no_smooth,
            posture_cost=args.posture_cost,
            snap_band_m=args.snap_band,
            min_drop_m=args.min_drop,
            surface_band_m=args.surface_band,
            snap_transition_frames=args.snap_transition_frames,
            approach_window_frames=args.approach_window,
            nontip_min_distance=args.nontip_min,
            nontip_detection_distance=args.nontip_detect,
            tip_min_distance=args.tip_min,
            tip_detection_distance=args.tip_detect,
        )


if __name__ == "__main__":
    main()
