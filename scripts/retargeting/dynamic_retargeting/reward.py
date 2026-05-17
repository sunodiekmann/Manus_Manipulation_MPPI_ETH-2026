"""Reward formulation for Stage 3 sampling MPC.

Pure-function batched reward, all inputs are numpy arrays. Per-rollout sum
over the horizon; higher reward = better (errors are negated).

    cube_pos    : -w_cp * ||cube_pos_sim − cube_pos_ref||         [m]
    cube_quat   : -w_cq * 2·acos(|⟨q_sim, q_ref⟩|)                [rad geodesic]
    finger_qpos : -w_fq * ||q_finger_sim − q_finger_ref||         [rad]
    contact     : -w_ct * Σ_f valid_h_f · ||tip_in_cube − target_cube||  [m]
                  (SPIDER virtual-contact constraint, relative cube frame)
    wrist_delta : -w_wd * ||wrist_axisangle_delta||               [rad]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import SLICE_WRIST


@dataclass
class RewardWeights:
    cube_pos: float = 1.0          # per metre
    cube_quat: float = 0.3         # per rad
    finger_qpos: float = 1.0       # per rad
    contact: float = 1.0           # per metre, summed over fingers
    wrist_delta: float = 0.5       # per rad
    qvel: float = 0.05             # per rad/s and m/s — encourages stable
                                   # equilibrium states (SPIDER's qvel_rew)
    cube_pos_deadband_m: float = 0.0
    """If > 0, the cube_pos error is computed in *wrist frame* (so it
    follows the hand) and the deadband-radius bubble around the reference
    is free (no penalty). Outside the bubble, the existing cube_pos weight
    applies to the surplus."""
    terminal_scale: float = 10.0
    """Multiplier on the LAST horizon-step reward. Forces MPC to care
    about where the cube ends up at horizon-end, not just per-step
    average. SPIDER uses 10-100; values >1 fix myopic drift where
    MPC accepts trajectories that look fine for 400ms but drift over
    seconds."""


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (wxyz). Broadcasts over leading dims of q."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v, axis=-1)
    return v + w * t + np.cross(qv, t, axis=-1)


def compute_horizon_reward(
    obs: dict,                                  # rollout output, all (N, H, ...)
    ref_qpos: np.ndarray,                       # (H, 27)
    ref_cube_pos: np.ndarray,                   # (H, 3)
    ref_cube_quat: np.ndarray,                  # (H, 4) wxyz
    ref_wrist_quat: np.ndarray,                 # (H, 4) — currently unused but kept for parity
    actions: np.ndarray,                        # (N, H, A) — wrist-delta penalty
    weights: RewardWeights,
    contact_target_per_step: np.ndarray | None, # (H, 5, 3) cube-frame, NaN = no target
) -> np.ndarray:
    """Per-sample cumulative reward, shape (N,)."""
    qpos = obs["qpos"]
    cube_pos = obs["cube_pos"]
    cube_quat = obs["cube_quat_wxyz"]
    tip_xpos = obs["tip_xpos"]
    qvel = obs.get("qvel")   # (N, H, nv) if env produced it, else None

    if weights.cube_pos_deadband_m > 0.0:
        # Compute the cube-position deviation in wrist frame so MPPI is
        # free to translate the cube along with the hand (the bubble moves
        # with the wrist). Then deadband: zero penalty inside `deadband_m`.
        delta_world = cube_pos - ref_cube_pos                          # (N, H, 3)
        inv_wrist_q = ref_wrist_quat * np.array([1.0, -1.0, -1.0, -1.0])
        delta_wrist = _quat_rotate_vec(inv_wrist_q[None, :, :], delta_world)
        pos_dist = np.linalg.norm(delta_wrist, axis=-1)                # (N, H)
        pos_err = np.maximum(0.0, pos_dist - weights.cube_pos_deadband_m)
    else:
        pos_err = np.linalg.norm(cube_pos - ref_cube_pos, axis=-1)     # (N, H)

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

    # Velocity penalty — encourages stable equilibrium states. Without this,
    # MPC accepts dynamically-OK-now trajectories that drift over seconds.
    if qvel is not None and weights.qvel > 0.0:
        qvel_mag = np.linalg.norm(qvel, axis=-1)                      # (N, H)
    else:
        qvel_mag = np.zeros_like(pos_err)

    per_step = (
        -weights.cube_pos * pos_err
        - weights.cube_quat * quat_err
        - weights.finger_qpos * finger_err
        - weights.contact * contact_term
        - weights.wrist_delta * wrist_pen
        - weights.qvel * qvel_mag
    )
    # Terminal step gets a larger weight — fix for myopic drift. MPC then
    # considers where the cube ends up at horizon-end, not just per-step.
    if weights.terminal_scale != 1.0:
        per_step[:, -1] *= float(weights.terminal_scale)
    return per_step.sum(axis=-1)
