"""Reward formulation for Stage 3 sampling MPC.

Pure-function batched reward, all inputs are numpy arrays. Per-rollout sum
over the horizon; higher reward = better (errors are negated).

    cube_pos    : -w_cp * ||cube_pos_sim − cube_pos_ref||         [m]
    cube_quat   : -w_cq * 2·acos(|⟨q_sim, q_ref⟩|)                [rad geodesic]
    finger_qpos : -w_fq * ||q_finger_sim − q_finger_ref||         [rad]
    contact     : -w_ct * Σ_f valid_h_f · ||tip_in_cube − target_cube||  [m]
                  (SPIDER virtual-contact constraint, relative cube frame)
    wrist_delta : -w_wd * ||wrist_axisangle_delta||               [rad]
    qvel        : -w_qv * ||finger_qvel_sim − finger_qvel_ref||   [rad/s]
                  (only finger DOFs — wrist is mocap-tracked, cube must be
                  free to rotate for in-hand-manipulation tasks)
    cube_angvel : -w_av * ||rotvec_sim − rotvec_ref||             [rad/tick]
                  per-tick cube rotation increment (∝ angular velocity;
                  the dt folds into the weight). Tracks the rotation
                  *motion* rather than the absolute orientation: the cube
                  may drift in absolute pose as long as it turns at the
                  reference rate/axis. Default weight 0 — opt in for a
                  task-style reward; pair with --w-cube-quat 0 to fully
                  switch from pose tracking to motion tracking (cube_pos
                  still anchors position). See dynamic_retargeting_status.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import SLICE_WRIST


@dataclass
class RewardWeights:
    # Defaults are the validated working config (cube_roll2 + cube_pitch2,
    # 2026-05-21): orientation-led tracking with a strong contact term and a
    # light finger_qpos pull. A bare Stage-3 run reproduces these results.
    cube_pos: float = 0.5          # per metre
    cube_quat: float = 0.7         # per rad
    finger_qpos: float = 0.2       # per rad — soft pull to IK ref; the IK
                                   # pose is "touch", not "grasp", so a
                                   # heavy weight here actively fights
                                   # force-closure discovery
    contact: float = 1.8           # per metre, summed over fingers
    wrist_delta: float = 0.3       # per rad
    qvel: float = 0.005            # per rad/s on finger qvel DEVIATION
                                   # from reference. Penalizing absolute
                                   # qvel (or wrist/cube qvel) blocks
                                   # in-hand rotation; restrict to finger
                                   # qvel deviation only.
    cube_angvel: float = 0.25       # per rad/tick on the cube per-tick
                                   # rotation increment vs reference.
                                   # Default 0 (pure pose tracking). Raise
                                   # it (and optionally drop cube_quat) for
                                   # motion/task-style tracking.
    terminal_scale: float = 3.0
    """Multiplier on the LAST horizon-step reward. Helps with myopic
    drift by making MPC care about where the cube ends up at horizon-end,
    not just per-step average. Kept modest — too high and the optimizer
    over-fits a step the noise schedule barely explores (with the SPIDER
    schedule, h=H-1 has the MOST noise, so the per-sample terminal
    reward signal is also the loudest; 3-5 is a reasonable range)."""


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (wxyz). Broadcasts over leading dims of q."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v, axis=-1)
    return v + w * t + np.cross(qv, t, axis=-1)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of wxyz quaternion(s). Broadcasts over leading dims."""
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for wxyz quaternions, broadcasting."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def rotvec_from_quat_seq(q: np.ndarray) -> np.ndarray:
    """Per-tick rotation vector (axis·angle) of a wxyz quaternion sequence.

    q: (..., H, 4) → (..., H, 3). For step h the increment is the world-frame
    rotation r = q[h+1] ⊗ conj(q[h]) converted to axis·angle; the last step
    repeats the previous (forward-difference padding). World-frame and method
    are identical for sim and reference, so ‖rotvec_sim − rotvec_ref‖ is a
    well-posed angular-velocity-tracking error (the per-tick dt folds into the
    reward weight)."""
    q0 = q[..., :-1, :]
    q1 = q[..., 1:, :]
    r = _quat_mul(q1, _quat_conj(q0))
    # Shortest-arc: force scalar part non-negative.
    r = np.where(r[..., 0:1] < 0.0, -r, r)
    w = np.clip(r[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)                                  # (..., H-1)
    s = np.sqrt(np.maximum(1.0 - w * w, 1e-12))
    axis = r[..., 1:] / s[..., None]
    rotvec = axis * angle[..., None]                            # (..., H-1, 3)
    return np.concatenate([rotvec, rotvec[..., -1:, :]], axis=-2)


TERM_NAMES = ("cube_pos", "cube_quat", "finger_qpos", "contact",
              "wrist_delta", "qvel", "cube_angvel")


def compute_horizon_reward(
    obs: dict,                                  # rollout output, all (N, H, ...)
    ref_qpos: np.ndarray,                       # (H, 27)
    ref_cube_pos: np.ndarray,                   # (H, 3)
    ref_cube_quat: np.ndarray,                  # (H, 4) wxyz
    ref_wrist_quat: np.ndarray,                 # (H, 4) — currently unused but kept for parity
    ref_finger_qvel: np.ndarray,                # (H, 20) — finite-diff of finger qpos ref
    actions: np.ndarray,                        # (N, H, A) — wrist-delta penalty
    weights: RewardWeights,
    contact_target_per_step: np.ndarray | None, # (H, 5, 3) cube-frame, NaN = no target
    ref_cube_rotvec: np.ndarray | None = None,  # (H, 3) — per-tick ref rotation increment
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns (total_per_sample, terms_per_sample).

    total_per_sample : (N,) cumulative reward across the horizon.
    terms_per_sample : dict[name → (N,)] — each weighted, terminal-scaled
        term's per-sample horizon sum. Sum of terms == total. Useful for
        diagnosing which term dominates and which the optimizer is making
        progress on across MPPI iterations."""
    qpos = obs["qpos"]
    cube_pos = obs["cube_pos"]
    cube_quat = obs["cube_quat_wxyz"]
    tip_xpos = obs["tip_xpos"]
    qvel = obs.get("qvel")   # (N, H, nv) if env produced it, else None

    pos_err = np.linalg.norm(cube_pos - ref_cube_pos, axis=-1)        # (N, H)

    dot = np.abs(np.sum(cube_quat * ref_cube_quat, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    quat_err = 2.0 * np.arccos(dot)                                   # (N, H)

    fingers_sim = qpos[..., 7:27]
    fingers_ref = ref_qpos[..., 7:27]
    finger_err = np.linalg.norm(fingers_sim - fingers_ref, axis=-1)   # (N, H)

    # Contact: SPIDER relative-position constraint in sim cube frame.
    rel_world = tip_xpos - cube_pos[:, :, None, :]                    # (N, H, 5, 3)
    inv_cube_quat = cube_quat * np.array([1.0, -1.0, -1.0, -1.0])
    rel_cube = _quat_rotate_vec(inv_cube_quat[:, :, None, :], rel_world)
    if contact_target_per_step is not None:
        valid = ~np.isnan(contact_target_per_step[..., 0])            # (H, 5)
        safe_target = np.where(valid[..., None], contact_target_per_step, 0.0)
        contact_err = np.linalg.norm(
            rel_cube - safe_target[None, :, :, :], axis=-1)           # (N, H, 5)
        contact_term = np.sum(
            contact_err * valid.astype(np.float64)[None, :, :], axis=-1)  # (N, H)
    else:
        contact_term = np.zeros_like(pos_err)

    wrist_delta = actions[..., SLICE_WRIST]
    wrist_pen = np.linalg.norm(wrist_delta, axis=-1)                  # (N, H)

    # Velocity penalty — encourages fingers to track the *reference*
    # qvel instead of being still (which would block in-hand rotation).
    # Wrist qvel is mocap-driven (penalty has nothing to optimize against)
    # and cube qvel should be free so MPC can discover regrip / pitch /
    # roll. We only penalize finger qvel deviation. dyn-scene qvel layout
    # is [6 wrist | 20 fingers | 6 cube], so finger qvel lives at 6:26.
    if qvel is not None and weights.qvel > 0.0:
        finger_qvel = qvel[..., 6:26]                                 # (N, H, 20)
        qvel_err = np.linalg.norm(
            finger_qvel - ref_finger_qvel[None, :, :], axis=-1)       # (N, H)
    else:
        qvel_err = np.zeros_like(pos_err)

    # Cube angular-velocity tracking: match the per-tick rotation increment
    # of the reference. Decouples absolute orientation (which may drift) from
    # the rotation *motion*. Computed by quaternion finite-difference on both
    # sim and ref (same world frame), so the error is well-posed.
    if weights.cube_angvel > 0.0 and ref_cube_rotvec is not None:
        sim_rotvec = rotvec_from_quat_seq(cube_quat)                  # (N, H, 3)
        angvel_err = np.linalg.norm(
            sim_rotvec - ref_cube_rotvec[None, :, :], axis=-1)        # (N, H)
    else:
        angvel_err = np.zeros_like(pos_err)

    terms_per_step = {
        "cube_pos":    -weights.cube_pos    * pos_err,
        "cube_quat":   -weights.cube_quat   * quat_err,
        "finger_qpos": -weights.finger_qpos * finger_err,
        "contact":     -weights.contact     * contact_term,
        "wrist_delta": -weights.wrist_delta * wrist_pen,
        "qvel":        -weights.qvel        * qvel_err,
        "cube_angvel": -weights.cube_angvel * angvel_err,
    }
    # Terminal step gets a larger weight — fix for myopic drift. MPC then
    # considers where the cube ends up at horizon-end, not just per-step.
    if weights.terminal_scale != 1.0:
        scale = float(weights.terminal_scale)
        for k in terms_per_step:
            terms_per_step[k] = terms_per_step[k].copy()
            terms_per_step[k][:, -1] *= scale
    terms = {k: v.sum(axis=-1) for k, v in terms_per_step.items()}
    total = sum(terms.values())
    return total, terms
