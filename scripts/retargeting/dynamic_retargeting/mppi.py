"""Sampling MPC, mirroring SPIDER's structure.

Per MPC tick we run `max_iters` of:
    1. Sample N noisy control sequences via knot-spline interpolation.
       - Per-component noise scales (finger / wrist / obj_pos / obj_rot
         decoupled).
       - Per-horizon-step log-spaced annealing
         (first_ctrl_noise_scale → last_ctrl_noise_scale).
       - Sample 0 is forced to zero noise so the prior plan is always
         evaluated; we never regress against last iter's plan.
    2. Roll all N samples out through `env.rollout_with_resample` (handles
       resample-on-drop, virtual-stick gain decay via env state).
    3. Score via `compute_horizon_reward` (per-step sums; lower-is-better
       errors are negated).
    4. Top-`top_frac` softmax → weighted plan update.
    5. Anneal global noise scale by `beta_traj` for the next iter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CpuEnv,
    N_ACTION,
    N_FINGER_CTRL,
    N_OBJ_POS_CTRL,
    N_OBJ_ROT_CTRL,
    N_WRIST_DELTA,
    SLICE_FINGER,
    SLICE_OBJ_POS,
    SLICE_OBJ_ROT,
    SLICE_WRIST,
)
from scripts.retargeting.dynamic_retargeting.reward import (
    RewardWeights,
    compute_horizon_reward,
)


@dataclass
class MPPIConfig:
    # Sampling budget
    horizon_steps: int = 8           # H, in ref-tick units
    num_samples: int = 256           # N (SPIDER's default 2048; CPU-tractable 256)
    num_knots: int = 4               # noise knots, interpolated to H
    max_iters: int = 8               # per-tick refinement iterations
    n_substeps: int = 25             # mj_step calls per ref tick (sim_dt=0.002, ref_dt=0.05)

    # Cross-entropy weighting
    top_frac: float = 0.1            # weight only the top-10% samples
    softmax_temp: float = 1.0

    # Per-iter global noise annealing
    beta_traj: float = 0.7

    # Per-horizon-step log-space annealing (SPIDER first_/last_ctrl_noise_scale).
    # noise_h(h) = exp( log(first) + (log(last) - log(first)) * h/(H-1) ).
    first_step_noise_scale: float = 1.0
    last_step_noise_scale: float = 0.3

    # Virtual-stick "training wheels" for the optimizer. Linearly annealed
    # ACROSS MPPI iterations within each MPC tick (matches SPIDER). Iter 0
    # rolls out with stick=iter_stick_start so MPPI quickly finds reward-
    # bearing actions; iter N rolls out with stick=iter_stick_end so the
    # converged plan must hold the cube with fingers alone. The world step
    # (apply plan[0]) always uses stick=0 — the stick never adds physics
    # force, it only seeds the search.
    iter_stick_start: float = 1.0
    iter_stick_end: float = 0.0
    iter_stick_schedule: str = "warm_then_off"   # "linear" or "warm_then_off"
    """If "linear": stick linearly anneals from iter_stick_start (iter 0) to
    iter_stick_end (iter max_iters-1). If "warm_then_off": iter 0 uses
    iter_stick_start; all subsequent iters use iter_stick_end. The latter
    gives MPC many iters of pure refinement against the converged stick
    setting instead of mostly refining stick-dependent plans."""

    # Per-component noise std (in ctrl units). With N=2048 samples and the
    # two-axis annealing (per-iter beta_traj=0.7, per-step log 1.0→0.3),
    # base noise can be aggressive — the late-iter / late-step samples
    # naturally clamp down (e.g. ×0.10 by iter 3 step 7), and the
    # zero-noise first sample preserves the prior. Finger noise needs to
    # span enough range that grip configurations can actually be explored.
    finger_noise_std: float = 0.15   # rad  (~9°) per joint per knot
    wrist_noise_std: float = 0.04    # rad axis-angle (~2.3°)
    obj_pos_noise_std: float = 0.0   # default OFF: with stick-off iters, this
    obj_rot_noise_std: float = 0.0   # ctrl has no force; with stick-on early
                                     # iters it just randomly nudges the cube.

    # Action clipping
    wrist_delta_clip: float = 0.05   # rad axis-angle
    obj_pos_delta_clip: float = 0.05 # m around the reference baseline
    obj_rot_delta_clip: float = 0.30 # rad around the reference baseline

    # Resample-on-drop threshold (None disables the swap inside rollout).
    drop_threshold_m: float | None = 0.10

    weights: RewardWeights = field(default_factory=RewardWeights)


def _interp_knots_to_horizon(knots: np.ndarray, H: int) -> np.ndarray:
    """Linear interpolation of (N, K, A) knots to (N, H, A) horizon."""
    N, K, A = knots.shape
    if K == H:
        return knots
    knot_t = np.linspace(0.0, H - 1, K)
    out = np.empty((N, H, A), dtype=knots.dtype)
    for h in range(H):
        if h <= knot_t[0]:
            out[:, h] = knots[:, 0]
            continue
        if h >= knot_t[-1]:
            out[:, h] = knots[:, -1]
            continue
        k1 = int(np.searchsorted(knot_t, h)) - 1
        k2 = k1 + 1
        alpha = (h - knot_t[k1]) / (knot_t[k2] - knot_t[k1])
        out[:, h] = (1 - alpha) * knots[:, k1] + alpha * knots[:, k2]
    return out


def _per_component_std(cfg: MPPIConfig) -> np.ndarray:
    """Build (N_ACTION,) std vector matching ctrl-dim semantics."""
    std = np.empty(N_ACTION, dtype=np.float64)
    std[SLICE_FINGER] = cfg.finger_noise_std
    std[SLICE_WRIST] = cfg.wrist_noise_std
    std[SLICE_OBJ_POS] = cfg.obj_pos_noise_std
    std[SLICE_OBJ_ROT] = cfg.obj_rot_noise_std
    return std


def _per_horizon_step_anneal(cfg: MPPIConfig, H: int) -> np.ndarray:
    """(H,) log-spaced multipliers from first → last step noise scale."""
    if H == 1:
        return np.asarray([cfg.first_step_noise_scale], dtype=np.float64)
    return np.logspace(
        np.log10(cfg.first_step_noise_scale),
        np.log10(cfg.last_step_noise_scale),
        num=H,
        base=10.0,
        dtype=np.float64,
    )


def _sample_actions(
    plan: np.ndarray,           # (H, N_ACTION)
    obj_baseline: np.ndarray,   # (H, N_OBJ_POS_CTRL + N_OBJ_ROT_CTRL) — the
                                # `ctrl-baseline` for obj dims; clipping is
                                # done relative to it
    cfg: MPPIConfig,
    global_noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample (N, H, N_ACTION) noisy controls around `plan`.

    Sample 0 is the unmodified plan (zero noise) — guarantees the prior is
    always evaluated this iteration.
    """
    H = plan.shape[0]
    N = cfg.num_samples
    base_std = _per_component_std(cfg)                          # (A,)
    knots_noise = rng.normal(0.0, 1.0, size=(N, cfg.num_knots, N_ACTION))
    delta_at_horizon = _interp_knots_to_horizon(knots_noise, H)  # (N, H, A)
    horizon_anneal = _per_horizon_step_anneal(cfg, H)            # (H,)
    delta = (
        delta_at_horizon
        * base_std[None, None, :]
        * horizon_anneal[None, :, None]
        * global_noise_scale
    )
    delta[0] = 0.0  # zero-noise prior evaluation

    samples = plan[None, :, :] + delta

    # Clip per-component to keep ctrl in sane ranges.
    samples[..., SLICE_WRIST] = np.clip(
        samples[..., SLICE_WRIST], -cfg.wrist_delta_clip, cfg.wrist_delta_clip)
    obj_pos_baseline = obj_baseline[:, :N_OBJ_POS_CTRL]
    obj_rot_baseline = obj_baseline[:, N_OBJ_POS_CTRL:]
    samples[..., SLICE_OBJ_POS] = np.clip(
        samples[..., SLICE_OBJ_POS],
        obj_pos_baseline[None, :, :] - cfg.obj_pos_delta_clip,
        obj_pos_baseline[None, :, :] + cfg.obj_pos_delta_clip,
    )
    samples[..., SLICE_OBJ_ROT] = np.clip(
        samples[..., SLICE_OBJ_ROT],
        obj_rot_baseline[None, :, :] - cfg.obj_rot_delta_clip,
        obj_rot_baseline[None, :, :] + cfg.obj_rot_delta_clip,
    )
    return samples


def _topk_softmax(rews: np.ndarray, top_frac: float, temp: float) -> np.ndarray:
    """SPIDER-style: rank by reward, weight only top-`top_frac` via softmax."""
    N = rews.shape[0]
    k = max(1, int(top_frac * N))
    nan_mask = np.isnan(rews) | np.isinf(rews)
    if nan_mask.all():
        return np.full(N, 1.0 / N, dtype=np.float64)
    rews = np.where(nan_mask, rews[~nan_mask].min(), rews)
    top_idx = np.argpartition(rews, -k)[-k:]
    weights = np.zeros(N, dtype=np.float64)
    top_rews = rews[top_idx]
    centred = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    e = np.exp(centred / temp)
    weights[top_idx] = e / e.sum()
    return weights


def optimize_step(
    env: CpuEnv,
    plan: np.ndarray,                 # (H, N_ACTION)
    obj_baseline: np.ndarray,         # (H, N_OBJ_CTRL) ref cube qpos this window
    ref_qpos: np.ndarray,             # (H, 27)
    ref_cube_pos: np.ndarray,         # (H, 3)
    ref_cube_quat: np.ndarray,        # (H, 4) wxyz
    ref_wrist_pos: np.ndarray,        # (H, 3)
    ref_wrist_quat: np.ndarray,       # (H, 4) wxyz
    contact_target_per_step: np.ndarray | None,  # (H, 5, 3) cube-frame, NaN where N/A
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """One MPC tick: refine `plan` via `cfg.max_iters` MPPI iterations.
    Snapshots the current world state into the batch at the start of every
    iteration (each iter's rollout starts from the same physical state).
    """
    info = {"rew_max": [], "rew_mean": [], "frac_terminated": [], "stick": []}
    for it in range(cfg.max_iters):
        global_scale = cfg.beta_traj ** it
        if cfg.iter_stick_schedule == "linear":
            denom = max(1, cfg.max_iters - 1)
            stick = cfg.iter_stick_start + (cfg.iter_stick_end - cfg.iter_stick_start) * (it / denom)
        elif cfg.iter_stick_schedule == "warm_then_off":
            stick = cfg.iter_stick_start if it == 0 else cfg.iter_stick_end
        else:
            raise ValueError(f"unknown iter_stick_schedule: {cfg.iter_stick_schedule}")
        env.set_obj_decay(stick)
        info["stick"].append(float(stick))
        samples = _sample_actions(plan, obj_baseline, cfg, global_scale, rng)

        env.snapshot_world_into_batch()
        obs = env.rollout_with_resample(
            actions=samples,                # may be modified by resample-on-drop
            ref_wrist_pos=ref_wrist_pos,
            ref_wrist_quat=ref_wrist_quat,
            n_substeps=cfg.n_substeps,
            ref_cube_pos=ref_cube_pos,
            drop_threshold_m=cfg.drop_threshold_m,
        )
        rews = compute_horizon_reward(
            obs=obs,
            ref_qpos=ref_qpos,
            ref_cube_pos=ref_cube_pos,
            ref_cube_quat=ref_cube_quat,
            ref_wrist_quat=ref_wrist_quat,
            actions=samples,
            weights=cfg.weights,
            contact_target_per_step=contact_target_per_step,
        )
        weights = _topk_softmax(rews, cfg.top_frac, cfg.softmax_temp)
        plan = (weights[:, None, None] * samples).sum(axis=0)

        info["rew_max"].append(float(rews.max()))
        info["rew_mean"].append(float(rews.mean()))
        info["frac_terminated"].append(float(obs["terminated"].mean()))
    return plan, info
