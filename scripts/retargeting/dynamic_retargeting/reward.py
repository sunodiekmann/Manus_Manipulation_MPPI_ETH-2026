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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import SLICE_WRIST


@dataclass
class RewardWeights:
    cube_pos: float = 1.0          # per metre
    cube_quat: float = 0.3         # per rad
    finger_qpos: float = 0.3       # per rad — soft pull to IK ref; the IK
                                   # pose is "touch", not "grasp", so a
                                   # heavy weight here actively fights
                                   # force-closure discovery
    contact: float = 1.0           # per metre, summed over fingers
    wrist_delta: float = 0.5       # per rad
    qvel: float = 0.005            # per rad/s on finger qvel DEVIATION
                                   # from reference. Penalizing absolute
                                   # qvel (or wrist/cube qvel) blocks
                                   # in-hand rotation; restrict to finger
                                   # qvel deviation only.
    cube_pos_deadband_m: float = 0.0
    """If > 0, the cube_pos error is computed in *wrist frame* (so it
    follows the hand) and the deadband-radius bubble around the reference
    is free (no penalty). Outside the bubble, the existing cube_pos weight
    applies to the surplus."""
    cube_pos_wall_start_m: float = 0.0
    """Distance *past the deadband* where the quadratic wall kicks in.
    Wall_scale only applies when pos_dist > deadband + wall_start. Set to
    e.g. 0.01 to leave a 10 mm linear-only band between the deadband and
    the wall (so MPC has gradient room to recover before hitting the wall)."""
    cube_pos_wall_scale: float = 0.0
    """Quadratic penalty weight (per m²) past `wall_start`. Multiplied by
    `cube_pos` like the linear term, so the effective penalty is
    `cube_pos * (linear_excess + wall_scale * hard_excess²)`. Use when
    "letting the cube drift" becomes a destabilising local optimum —
    e.g. wall_scale=100 means at 70 mm beyond wall_start the quadratic
    contribution per step is 0.49, dominating the linear term (0.07)."""
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


TERM_NAMES = ("cube_pos", "cube_quat", "finger_qpos", "contact",
              "wrist_delta", "qvel")


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

    # Use wrist-frame distance whenever either deadband or wall is active
    # so the bubble + wall move with the hand.
    use_wrist_frame = (weights.cube_pos_deadband_m > 0.0
                       or weights.cube_pos_wall_scale > 0.0)
    if use_wrist_frame:
        delta_world = cube_pos - ref_cube_pos                          # (N, H, 3)
        inv_wrist_q = ref_wrist_quat * np.array([1.0, -1.0, -1.0, -1.0])
        delta_wrist = _quat_rotate_vec(inv_wrist_q[None, :, :], delta_world)
        pos_dist = np.linalg.norm(delta_wrist, axis=-1)                # (N, H)
    else:
        pos_dist = np.linalg.norm(cube_pos - ref_cube_pos, axis=-1)    # (N, H)

    # Linear part: zero inside deadband, linear in excess outside.
    pos_err = np.maximum(0.0, pos_dist - weights.cube_pos_deadband_m)
    # Quadratic "hard wall" past deadband + wall_start. Drowns out small
    # gradient gains the optimizer might claim by letting the cube drift
    # past a recoverable distance — once it's too far, no MPC plan can
    # recover the grip geometrically.
    if weights.cube_pos_wall_scale > 0.0:
        wall_thresh = weights.cube_pos_deadband_m + weights.cube_pos_wall_start_m
        hard_excess = np.maximum(0.0, pos_dist - wall_thresh)
        pos_err = pos_err + weights.cube_pos_wall_scale * hard_excess ** 2

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

    terms_per_step = {
        "cube_pos":    -weights.cube_pos    * pos_err,
        "cube_quat":   -weights.cube_quat   * quat_err,
        "finger_qpos": -weights.finger_qpos * finger_err,
        "contact":     -weights.contact     * contact_term,
        "wrist_delta": -weights.wrist_delta * wrist_pen,
        "qvel":        -weights.qvel        * qvel_err,
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
