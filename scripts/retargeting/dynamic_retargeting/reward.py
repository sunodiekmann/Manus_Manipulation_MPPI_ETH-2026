"""Reward formulation for dynamic retargeting MPPI.

All inputs are numpy arrays; pure functions, no MuJoCo state. This makes the
reward portable to any backend (CPU, MJX) and unit-testable.

Reward (higher is better) per rollout sample, summed over horizon. All terms
use **L2 norms** (linear in error magnitude), matching SPIDER's reward, so
weight tuning is intuitive and large errors aren't double-penalised. The
contact term follows SPIDER's virtual-contact-guidance formulation (Eq. 4):
it is a *relative* constraint between tip and cube, expressed in the sim
cube's frame, so cube tracking and grip can no longer fight each other.

    cube_pos     :  -w_cp  * ||cube_pos_sim − cube_pos_ref||      [m]
    cube_quat    :  -w_cq  * 2·acos(|⟨q_sim, q_ref⟩|)              [rad geodesic]
    finger_qpos  :  -w_fq  * ||q_finger_sim − q_finger_ref||       [rad]
    contact      :  -w_ct  * Σ_f valid[f]·mask[f]·||tip_in_cube_f − target_cube_f||  [m]
    wrist_delta  :  -w_wd  * ||wrist_delta_axis_angle||            [rad]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    """Defaults match SPIDER's no-contact preset (pos=1.0, rot=0.3)."""
    cube_pos: float = 1.0        # per metre — 1 cm cube drift = 0.01
    cube_quat: float = 0.3       # per rad   — 30° drift ≈ 0.16
    finger_qpos: float = 1.0     # per rad   — 20-joint 0.2 rad uniform drift ≈ 0.89
    contact: float = 0.0         # per metre, summed over fingers — off by default
    wrist_delta: float = 0.5     # per rad   — discourages wrist twisting away from ref
    contact_always_on: bool = False
    """If True, contact reward pulls every valid-target finger toward its
    contact_target every frame, ignoring `contact_mask`. Useful when the
    human recording under-engages some fingers (e.g. index/little) but you
    want the robot to keep contact for grip stability."""


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by q (wxyz). Broadcasts over any leading dims of q."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * np.cross(qv, v, axis=-1)
    return v + w * t + np.cross(qv, t, axis=-1)


def compute_horizon_reward(
    obs: dict,                            # batch-rollout output, all (N, H, ...)
    ref_qpos: np.ndarray,                 # (H, 27)  hand qpos reference
    ref_cube_pos: np.ndarray,             # (H, 3)
    ref_cube_quat: np.ndarray,            # (H, 4)   wxyz
    ref_wrist_quat: np.ndarray,           # (H, 4)   wxyz (the unperturbed reference)
    contact_mask: np.ndarray,             # (5,) bool — legacy whole-finger gate (ignored if per-frame target given)
    contact_target_cube: np.ndarray,      # (5, 3) legacy single target per finger (used if per-frame not given)
    contact_target_valid: np.ndarray,     # (5,) bool — legacy per-finger validity
    actions: np.ndarray,                  # (N, H, 23) — for wrist-delta penalty
    weights: RewardWeights,
    contact_target_per_step: np.ndarray | None = None,  # (H, 5, 3), NaN where no target
) -> np.ndarray:
    """Returns cumulative reward, shape (N,)."""
    N, H = obs["qpos"].shape[:2]

    # cube position — L2 norm in metres
    cube_pos = obs["cube_pos"]                          # (N, H, 3)
    pos_err = np.linalg.norm(cube_pos - ref_cube_pos, axis=-1)  # (N, H)

    # cube orientation — geodesic angle (radians), 0 = aligned, π = flipped
    cube_quat = obs["cube_quat"]                        # (N, H, 4)
    dot = np.abs(np.sum(cube_quat * ref_cube_quat, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    quat_err = 2.0 * np.arccos(dot)                     # (N, H), in [0, π]

    # finger qpos tracking — L2 norm in radians
    qpos = obs["qpos"]                                  # (N, H, 27 + 7)
    fingers_sim = qpos[..., 7:27]
    fingers_ref = ref_qpos[..., 7:27]
    finger_err = np.linalg.norm(fingers_sim - fingers_ref, axis=-1)  # (N, H)

    # Per-finger contact reward — SPIDER-style relative-position constraint
    # (paper Eq. 4): express the sim tip in the SIM cube's frame and compare
    # to the cube-frame target. This decouples grip from cube tracking: as the
    # cube moves/rotates, the constraint moves with it, so contact and cube_pos
    # no longer fight each other.
    tip_xpos = obs["tip_xpos"]                          # (N, H, 5, 3)
    rel_world = tip_xpos - cube_pos[:, :, None, :]      # (N, H, 5, 3)
    # Inverse of unit quaternion = conjugate (negate xyz, keep w).
    inv_cube_quat = cube_quat * np.array([1.0, -1.0, -1.0, -1.0])
    rel_cube_frame = quat_rotate_vec(
        inv_cube_quat[:, :, None, :], rel_world
    )                                                   # (N, H, 5, 3)

    if contact_target_per_step is not None:
        # Per-frame target: NaN signals "no target this finger at this frame".
        # Compute the (H, 5) validity mask from NaN-ness, replace NaNs with the
        # rel_cube_frame value (so err = 0 there and they contribute nothing).
        per_step_valid = ~np.isnan(contact_target_per_step[..., 0])  # (H, 5)
        # Broadcast-safe fill of NaN with a value that gives 0 error.
        safe_target = np.where(
            per_step_valid[..., None], contact_target_per_step, 0.0
        )                                               # (H, 5, 3)
        contact_err = np.linalg.norm(
            rel_cube_frame - safe_target[None, :, :, :], axis=-1
        )                                               # (N, H, 5)
        weight_per_step = per_step_valid.astype(np.float64)[None, :, :]  # (1, H, 5)
        contact_term = np.sum(contact_err * weight_per_step, axis=-1)    # (N, H)
    else:
        # Legacy single-target-per-finger path.
        contact_err = np.linalg.norm(
            rel_cube_frame - contact_target_cube[None, None, :, :], axis=-1
        )                                               # (N, H, 5)
        if weights.contact_always_on:
            weight_per_finger = contact_target_valid.astype(np.float64)
        else:
            weight_per_finger = (contact_mask & contact_target_valid).astype(np.float64)
        contact_term = np.sum(contact_err * weight_per_finger[None, None, :], axis=-1)

    # wrist-delta penalty — L2 of axis-angle (already a rotation magnitude)
    wrist_delta = actions[..., 20:23]
    wrist_pen = np.linalg.norm(wrist_delta, axis=-1)    # (N, H)

    per_step = (
        -weights.cube_pos * pos_err
        - weights.cube_quat * quat_err
        - weights.finger_qpos * finger_err
        - weights.contact * contact_term
        - weights.wrist_delta * wrist_pen
    )
    return per_step.sum(axis=-1)
