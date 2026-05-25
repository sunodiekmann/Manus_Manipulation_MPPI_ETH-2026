"""Sampling MPC, mirroring SPIDER's structure.

Per MPC tick we run `max_iters` of:
    1. Sample N noisy control sequences via knot-spline interpolation.
       - Per-component noise scales (finger / wrist decoupled).
       - Per-horizon-step log-spaced annealing
         (first_step_noise_scale → last_step_noise_scale; SPIDER convention
         is LOW noise near term, HIGH noise far term so plan[0] converges
         tightly between iters and the unknown future stays exploratory).
       - Sample 0 is forced to zero noise so the prior plan is always
         evaluated; we never regress against last iter's plan.
    2. Roll all N samples out through `env.rollout_with_resample`.
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
    N_WRIST_DELTA,
    SLICE_FINGER,
    SLICE_WRIST,
)
from scripts.retargeting.dynamic_retargeting.reward import (
    RewardWeights,
    TERM_NAMES,
    compute_horizon_reward,
    rotvec_from_quat_seq,
)


@dataclass
class MPPIConfig:
    # Sampling budget. Defaults are SPIDER-matched (their Config + mjwp example):
    # 4 control steps over a 1.6s horizon, one knot each (no interpolation),
    # 40 sim-substeps per control step at sim_dt=0.01 → ctrl_dt=0.4s.
    horizon_steps: int = 4           # H control steps (SPIDER horizon/ctrl_dt = 1.6/0.4)
    num_samples: int = 1024          # N (SPIDER example 1024; Config default 2048)
    num_knots: int = 4               # control knots; == H here → no interpolation
    max_iters: int = 16              # per-step refinement iters (SPIDER, early-stoppable)
    n_substeps: int = 40             # mj_step per control step (ctrl_dt/sim_dt = 0.4/0.01)
    sim_dt: float = 0.01             # SPIDER setup_mj_model timestep

    # Cross-entropy weighting. Lower temp = sharper weighting on top samples.
    top_frac: float = 0.1            # weight only the top-10% samples
    softmax_temp: float = 0.1        # SPIDER temperature

    # Per-iter global noise annealing. SPIDER computes beta from final_noise_scale
    # so that scale**max_iters == final_noise_scale (set in optimize_step).
    final_noise_scale: float = 0.1   # SPIDER final_noise_scale
    beta_traj: float = 0.7           # fallback if final_noise_scale <= 0

    # Per-horizon-step log-space annealing (SPIDER first_/last_ctrl_noise_scale).
    # LOW noise on the near-term step, HIGH noise on the far horizon end.
    first_step_noise_scale: float = 0.5   # SPIDER first_ctrl_noise_scale
    last_step_noise_scale: float = 1.0    # SPIDER last_ctrl_noise_scale

    # Per-component noise std (in ctrl units). SPIDER joint 0.15, pos/rot 0.03.
    finger_noise_std: float = 0.15   # rad per joint per knot
    wrist_noise_std: float = 0.03    # rad axis-angle

    # Action clipping
    wrist_delta_clip: float = 0.05   # rad axis-angle

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
    cfg: MPPIConfig,
    global_noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample (N, H, N_ACTION) noisy controls around `plan`. Sample 0 is the
    unmodified plan (zero noise) — guarantees the prior is always
    evaluated this iteration."""
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

    samples[..., SLICE_WRIST] = np.clip(
        samples[..., SLICE_WRIST], -cfg.wrist_delta_clip, cfg.wrist_delta_clip)
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
    # Finger qvel reference for the qvel deviation penalty: per-horizon-step
    # finite diff of the finger qpos reference. Skips wrist (mocap-driven)
    # and cube (penalizing cube velocity blocks in-hand rotation).
    ref_dt = cfg.n_substeps * cfg.sim_dt
    ref_finger_qvel = np.zeros_like(ref_qpos[..., 7:27])
    if ref_qpos.shape[0] > 1:
        ref_finger_qvel[:-1] = (ref_qpos[1:, 7:27] - ref_qpos[:-1, 7:27]) / ref_dt
        ref_finger_qvel[-1] = ref_finger_qvel[-2]

    # Per-tick reference cube rotation increment for the cube_angvel term.
    # Only needed when that term is active. dt folds into the weight, so this
    # is the raw per-tick rotation vector (not divided by ref_dt).
    ref_cube_rotvec = (
        rotvec_from_quat_seq(ref_cube_quat) if cfg.weights.cube_angvel > 0.0
        else None
    )

    info = {
        "rew_max": [], "rew_mean": [], "frac_terminated": [],
        "terms": {k: [] for k in TERM_NAMES},
    }
    # SPIDER global-noise anneal: beta s.t. beta**max_iters == final_noise_scale.
    if cfg.final_noise_scale > 0.0 and cfg.max_iters > 0:
        beta = cfg.final_noise_scale ** (1.0 / cfg.max_iters)
    else:
        beta = cfg.beta_traj
    for it in range(cfg.max_iters):
        global_scale = beta ** it
        samples = _sample_actions(plan, cfg, global_scale, rng)

        env.snapshot_world_into_batch()
        obs = env.rollout_with_resample(
            actions=samples,                # may be modified by resample-on-drop
            ref_wrist_pos=ref_wrist_pos,
            ref_wrist_quat=ref_wrist_quat,
            n_substeps=cfg.n_substeps,
            ref_cube_pos=ref_cube_pos,
            drop_threshold_m=cfg.drop_threshold_m,
        )
        rews, term_per_sample = compute_horizon_reward(
            obs=obs,
            ref_qpos=ref_qpos,
            ref_cube_pos=ref_cube_pos,
            ref_cube_quat=ref_cube_quat,
            ref_wrist_quat=ref_wrist_quat,
            ref_finger_qvel=ref_finger_qvel,
            actions=samples,
            weights=cfg.weights,
            contact_target_per_step=contact_target_per_step,
            ref_cube_rotvec=ref_cube_rotvec,
        )
        weights = _topk_softmax(rews, cfg.top_frac, cfg.softmax_temp)
        plan = (weights[:, None, None] * samples).sum(axis=0)
        # Re-clip wrist after weighted average
        plan[:, SLICE_WRIST] = np.clip(
            plan[:, SLICE_WRIST], -cfg.wrist_delta_clip, cfg.wrist_delta_clip)

        info["rew_max"].append(float(rews.max()))
        info["rew_mean"].append(float(rews.mean()))
        info["frac_terminated"].append(float(obs["terminated"].mean()))
        for k, v in term_per_sample.items():
            info["terms"][k].append(float(v.mean()))
    return plan, info
