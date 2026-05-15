"""Stage 2c, Pass 1 — sequence-wide cube pose smoothing.

Treats the cube pose as a decision variable around the recorded pose:
    new_cube_pose[t] = recorded_cube_pose[t] ⊕ δ[t]
where δ[t] = (δ_pos[t], δ_rot_aa[t]) ∈ R^6, bounded by ±pos_bound_m / ±rot_bound_rad.

Loss (smooth, sequence-wide):
    contact:  for each in-contact (t, f), pull the cube so the recorded human
              fingertip (in cube frame) ends up *near* the actual robot tip
              from the Stage 2a IK. Effectively asks the cube to absorb the
              residual robot-vs-human geometry gap, frame by frame.
    smooth:   acceleration penalty on δ — penalises jitter, not the recorded
              motion itself.
    zero:     weak L2 bias toward δ=0 so unconstrained frames default to
              "use the recorded pose".

Solver: scipy L-BFGS-B with JAX-jit'd value + grad. Box bounds elementwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.tesollo_constants import FINGERS, TARGET_SCALE
from scripts.utils.transforms import pose_to_matrix


# ── precompute ───────────────────────────────────────────────────────────────

@dataclass
class SmootherInputs:
    """JAX-side constants for the smoothing problem.

    Stacked over N frames × 5 fingers. NaN-targets (no contact this frame, f)
    are flagged by contact_weight=0 so they contribute zero loss.
    """
    recorded_cube_pos: jnp.ndarray   # (N, 3)
    rotated_offset: jnp.ndarray      # (N, 5, 3) — R_world_cube @ target_cube
    tip_world: jnp.ndarray           # (N, 5, 3) — robot tip world pos (FK)
    contact_weight: jnp.ndarray      # (N, 5) — 1.0 when in contact, else 0.0


def _fk_tip_positions(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,            # (N, model.nq) but may be 27; pad as needed
    cube_pose_seq: np.ndarray,        # (N, 7)
) -> np.ndarray:
    """Forward-kinematics every frame and return the 5 tip world positions.

    Skips frames where qpos has NaN — returns NaN tips for those frames.
    """
    N = qpos_seq.shape[0]
    data = mujoco.MjData(model)
    tip_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FINGERS
    ]
    if any(g < 0 for g in tip_geom_ids):
        missing = [f for f, g in zip(FINGERS, tip_geom_ids) if g < 0]
        raise ValueError(f"missing tip geoms: {missing}")

    out = np.full((N, 5, 3), np.nan)
    nq = model.nq
    nq_hand = qpos_seq.shape[1]
    for t in range(N):
        q = qpos_seq[t]
        if np.any(np.isnan(q)):
            continue
        data.qpos[:nq_hand] = q
        if nq > nq_hand:
            # Hand-only nq=27 + cube freejoint (7) → 34. Populate cube qpos too
            # so collisions / FK chain are consistent (FK only needs the hand
            # chain but we set it anyway for completeness).
            cp = cube_pose_seq[t]
            if cp is not None and not np.any(np.isnan(cp)):
                data.qpos[nq_hand : nq_hand + 7] = cp
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for f, gid in enumerate(tip_geom_ids):
            out[t, f] = data.geom_xpos[gid]
    return out


def prepare_inputs(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,                          # (N, 27) cube_relative IK
    cube_pose_seq: np.ndarray,                      # (N, 7) recorded cube pose
    intent_target_per_frame: np.ndarray,             # (N, 5, 3) cube-frame, NaN where no intent
) -> SmootherInputs:
    """Build the JAX-side constants from the Stage 2a output + intent targets.

    `intent_target_per_frame` is the cube-frame point that the caller asserts
    *should* line up with the robot tip at each (t, f), or NaN to skip that
    pair. Both the segmented contact target (from spider_contact) and the raw
    (offset+scale-adjusted) recorded fingertip work as sources; the smoother
    treats them uniformly.
    """
    N = qpos_seq.shape[0]
    cube_pose_np = np.asarray(cube_pose_seq, dtype=float)

    # FK every frame on the cube_relative qpos.
    tip_world = _fk_tip_positions(model, qpos_seq, cube_pose_np)   # (N, 5, 3)

    recorded_pos = cube_pose_np[:, :3]                                # (N, 3)
    quat_wxyz = cube_pose_np[:, 3:7]                                  # (N, 4)
    # Convert wxyz → xyzw for scipy Rotation, then build rotation matrices.
    quat_xyzw = np.stack(
        [quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]],
        axis=-1,
    )
    valid_pose = ~np.any(np.isnan(quat_xyzw), axis=-1)
    R_world_cube = np.zeros((N, 3, 3))
    R_world_cube[valid_pose] = R.from_quat(quat_xyzw[valid_pose]).as_matrix()

    intent = np.asarray(intent_target_per_frame, dtype=float)

    rotated_offset = np.zeros((N, 5, 3))
    contact_weight = np.zeros((N, 5))
    for t in range(N):
        if not valid_pose[t]:
            continue
        for f in range(5):
            tgt = intent[t, f]
            if np.any(np.isnan(tgt)) or np.any(np.isnan(tip_world[t, f])):
                continue
            rotated_offset[t, f] = R_world_cube[t] @ tgt
            contact_weight[t, f] = 1.0

    return SmootherInputs(
        recorded_cube_pos=jnp.asarray(recorded_pos),
        rotated_offset=jnp.asarray(rotated_offset),
        tip_world=jnp.asarray(tip_world),
        contact_weight=jnp.asarray(contact_weight),
    )


# ── loss ─────────────────────────────────────────────────────────────────────

def _axis_angle_to_matrix(aa: jnp.ndarray) -> jnp.ndarray:
    """(..., 3) axis-angle (= axis × angle) → (..., 3, 3) rotation matrix.

    Rodrigues formula written so that gradients through angle=0 are finite.
    Uses Taylor expansion of sin(θ)/θ and (1-cos θ)/θ² when θ² is small, the
    closed form otherwise. The "double-where" pattern feeds a safe θ²>0 into
    the closed-form branch even when θ²=0 is the selected branch, preventing
    NaN gradients via jnp.where (see JAX FAQ on NaN-safe where).
    """
    # Skew-symmetric matrix of aa (no normalization).
    zero = jnp.zeros_like(aa[..., 0])
    x, y, z = aa[..., 0], aa[..., 1], aa[..., 2]
    K = jnp.stack([
        jnp.stack([zero, -z, y], axis=-1),
        jnp.stack([z, zero, -x], axis=-1),
        jnp.stack([-y, x, zero], axis=-1),
    ], axis=-2)                                                      # (..., 3, 3)
    K2 = jnp.matmul(K, K)

    theta_sq = jnp.sum(aa * aa, axis=-1)                              # (...,)
    SMALL = 1e-8
    use_taylor = theta_sq < SMALL
    # Safe value for the closed-form branch when θ² is small.
    safe_sq = jnp.where(use_taylor, 1.0, theta_sq)
    safe_theta = jnp.sqrt(safe_sq)
    a_closed = jnp.sin(safe_theta) / safe_theta                       # sin(θ)/θ
    b_closed = (1.0 - jnp.cos(safe_theta)) / safe_sq                  # (1-cos θ)/θ²

    t = theta_sq
    a_taylor = 1.0 - t / 6.0 + t * t / 120.0 - t ** 3 / 5040.0
    b_taylor = 0.5 - t / 24.0 + t * t / 720.0 - t ** 3 / 40320.0

    a = jnp.where(use_taylor, a_taylor, a_closed)
    b = jnp.where(use_taylor, b_taylor, b_closed)

    eye = jnp.broadcast_to(jnp.eye(3), K.shape)
    return eye + a[..., None, None] * K + b[..., None, None] * K2


def _loss_components(
    delta_flat: jnp.ndarray,         # (N * 6,)
    inputs: SmootherInputs,
    lambda_smooth: float,
    lambda_zero: float,
):
    N = inputs.recorded_cube_pos.shape[0]
    delta = delta_flat.reshape(N, 6)
    delta_pos = delta[:, :3]                                           # (N, 3)
    delta_rot_aa = delta[:, 3:]                                        # (N, 3)
    R_delta = _axis_angle_to_matrix(delta_rot_aa)                      # (N, 3, 3)

    # new_target_world[t, f] = recorded_cube_pos[t] + delta_pos[t] + R_delta[t] @ offset[t, f]
    rotated = jnp.einsum("tij,tfj->tfi", R_delta, inputs.rotated_offset)   # (N, 5, 3)
    new_target_world = (
        inputs.recorded_cube_pos[:, None, :]
        + delta_pos[:, None, :]
        + rotated
    )                                                                    # (N, 5, 3)

    err = new_target_world - inputs.tip_world                            # (N, 5, 3)
    err_sq = jnp.sum(err * err, axis=-1)                                  # (N, 5)
    contact = jnp.sum(inputs.contact_weight * err_sq)

    # Smoothness: acceleration penalty on δ (||δ[t-1] - 2δ[t] + δ[t+1]||²).
    if N >= 3:
        accel = delta[2:] - 2.0 * delta[1:-1] + delta[:-2]               # (N-2, 6)
        smooth = jnp.sum(accel * accel)
    else:
        smooth = jnp.array(0.0)

    zero = jnp.sum(delta * delta)

    return contact, smooth, zero


def _loss(delta_flat, inputs, lambda_smooth, lambda_zero):
    c, s, z = _loss_components(delta_flat, inputs, lambda_smooth, lambda_zero)
    return c + lambda_smooth * s + lambda_zero * z


# ── solver ───────────────────────────────────────────────────────────────────

def smooth_cube_pose(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    cube_pose_seq: np.ndarray,
    intent_target_per_frame: np.ndarray,
    pos_bound_m: float = 0.005,
    rot_bound_rad: float = np.deg2rad(5.0),
    lambda_smooth: float = 100.0,
    lambda_zero: float = 1.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Solve the sequence-wide cube pose smoothing problem.

    Returns
    -------
    smoothed_cube_pose : (N, 7)  — [x,y,z,qw,qx,qy,qz] per frame
    info : dict — diagnostics (final loss components, init delta norm, etc.)
    """
    N = qpos_seq.shape[0]
    inputs = prepare_inputs(model, qpos_seq, cube_pose_seq, intent_target_per_frame)
    in_contact_frames = int(np.asarray(inputs.contact_weight > 0).sum())
    if verbose:
        print(f"  smoother: N={N} frames, {in_contact_frames} (t,f) contact terms,"
              f" bounds ±{pos_bound_m*1000:.1f}mm / ±{np.rad2deg(rot_bound_rad):.1f}°")

    if in_contact_frames == 0:
        # Degenerate: no contact pulls — best δ is zero. Skip optimization.
        if verbose:
            print("  smoother: no contact terms → returning recorded cube pose")
        return np.asarray(cube_pose_seq, dtype=float).copy(), {"skipped": True}

    val_and_grad = jax.jit(jax.value_and_grad(
        lambda x: _loss(x, inputs, lambda_smooth, lambda_zero)
    ))

    def fg(x_np):
        v, g = val_and_grad(jnp.asarray(x_np))
        return float(v), np.asarray(g, dtype=np.float64)

    x0 = np.zeros(N * 6, dtype=np.float64)
    bounds = []
    for _ in range(N):
        bounds += [(-pos_bound_m, pos_bound_m)] * 3
        bounds += [(-rot_bound_rad, rot_bound_rad)] * 3

    result = minimize(
        fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-7, "disp": verbose},
    )

    delta = np.asarray(result.x).reshape(N, 6)
    if verbose:
        c, s, z = _loss_components(jnp.asarray(result.x), inputs,
                                   lambda_smooth, lambda_zero)
        pos_max = float(np.max(np.linalg.norm(delta[:, :3], axis=-1)) * 1000)
        rot_max = float(np.rad2deg(np.max(np.linalg.norm(delta[:, 3:], axis=-1))))
        print(f"  smoother: loss contact={float(c)*1e6:.2f} (mm²), "
              f"smooth={float(s):.3e}, zero={float(z):.3e}; "
              f"max |δ_pos|={pos_max:.2f} mm, max |δ_rot|={rot_max:.2f}°")

    smoothed = _apply_delta_to_pose_seq(cube_pose_seq, delta)
    info = {
        "delta": delta,
        "loss_total": float(result.fun),
        "n_iter": int(result.nit),
        "success": bool(result.success),
    }
    return smoothed, info


def _apply_delta_to_pose_seq(cube_pose_seq: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply per-frame (δ_pos, δ_rot_aa) to recorded cube pose.

    cube_pose_seq: (N, 7) [x,y,z, qw,qx,qy,qz]
    delta:        (N, 6) [dx,dy,dz, ax,ay,az] (axis-angle, world frame)
    Returns:      (N, 7)
    """
    N = cube_pose_seq.shape[0]
    out = np.asarray(cube_pose_seq, dtype=float).copy()
    delta_pos = delta[:, :3]
    delta_rot_aa = delta[:, 3:]
    valid = ~np.any(np.isnan(out), axis=-1)
    for t in range(N):
        if not valid[t]:
            continue
        out[t, :3] = out[t, :3] + delta_pos[t]
        # R_new = R_delta @ R_recorded (left-multiplication = world-frame rotation)
        recorded_xyzw = np.array([out[t, 4], out[t, 5], out[t, 6], out[t, 3]])
        R_rec = R.from_quat(recorded_xyzw)
        R_delta = R.from_rotvec(delta_rot_aa[t])
        R_new = R_delta * R_rec
        new_xyzw = R_new.as_quat()
        out[t, 3] = new_xyzw[3]   # qw
        out[t, 4] = new_xyzw[0]
        out[t, 5] = new_xyzw[1]
        out[t, 6] = new_xyzw[2]
    return out
