"""DIAL-MPC sampling-based MPC, mirroring SPIDER's optimizer structure but
stripped down to a single-backend, single-task implementation.

At each MPC tick we run `max_iters` of:
    1. Sample N noisy ctrl sequences via knot-spline interpolation.
    2. Roll out each through env.rollout_batch, score with reward.compute_horizon_reward.
    3. Top-k softmax weighting → updated plan.
    4. Anneal noise scale by beta_traj.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CpuEnv,
    N_ACTION,
    N_FINGER_CTRL,
    N_WRIST_DELTA,
)
from scripts.retargeting.dynamic_retargeting.reward import (
    RewardWeights,
    compute_horizon_reward,
)


@dataclass
class MPPIConfig:
    horizon_steps: int = 8         # H, in ref-tick units
    num_samples: int = 32          # N
    num_knots: int = 4             # noise knots, interpolated to H
    max_iters: int = 2
    top_frac: float = 0.1          # top-10% weighted softmax
    softmax_temp: float = 1.0
    beta_traj: float = 0.7         # noise annealing per iter
    n_substeps: int = 25           # mj_step calls per ref tick (sim_dt=0.002, ref_dt=0.05)
    finger_noise_std: float = 0.02 # rad (~1.1°); larger values destabilise gripping
    wrist_noise_std: float = 0.005 # rad axis-angle (per knot, before annealing)
    wrist_delta_clip: float = 0.05 # rad (~3°), prevent wrist twists from breaking grip
    weights: RewardWeights = field(default_factory=RewardWeights)


def _interp_knots_to_horizon(knots: np.ndarray, H: int) -> np.ndarray:
    """Linear interpolation of (N, K, A) knots to (N, H, A) horizon."""
    N, K, A = knots.shape
    if K == H:
        return knots
    # Position each knot at K-1 evenly spaced indices in [0, H-1]
    knot_t = np.linspace(0, H - 1, K)
    out = np.empty((N, H, A), dtype=knots.dtype)
    for h in range(H):
        # find bracketing knots
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


def _sample_actions(
    plan: np.ndarray,            # (H, 23)
    cfg: MPPIConfig,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Returns (N, H, 23) noisy controls around the plan."""
    H = plan.shape[0]
    std = np.empty(N_ACTION, dtype=np.float64)
    std[:N_FINGER_CTRL] = cfg.finger_noise_std
    std[N_FINGER_CTRL:] = cfg.wrist_noise_std
    knots = rng.normal(
        0.0, std, size=(cfg.num_samples, cfg.num_knots, N_ACTION)
    ) * noise_scale
    delta = _interp_knots_to_horizon(knots, H)
    samples = plan[None, :, :] + delta
    # Clip wrist axis-angle to keep it sane
    samples[..., N_FINGER_CTRL:] = np.clip(
        samples[..., N_FINGER_CTRL:],
        -cfg.wrist_delta_clip,
        cfg.wrist_delta_clip,
    )
    return samples


def _topk_softmax(rews: np.ndarray, top_frac: float, temp: float) -> np.ndarray:
    """Top-k softmax: only the top `top_frac` of samples get nonzero weight."""
    N = rews.shape[0]
    k = max(1, int(top_frac * N))
    top_idx = np.argpartition(rews, -k)[-k:]
    weights = np.zeros(N, dtype=np.float64)
    top_rews = rews[top_idx]
    centred = (top_rews - top_rews.mean()) / (top_rews.std() + 1e-2)
    e = np.exp(centred / temp)
    weights[top_idx] = e / e.sum()
    return weights


def make_cpu_score_fn(
    env: CpuEnv,
    cfg: MPPIConfig,
    ref_qpos, ref_cube_pos, ref_cube_quat, ref_wrist_pos, ref_wrist_quat,
    contact_mask, contact_target_cube, contact_target_valid,
    contact_target_per_step=None,
):
    """Closure that runs N rollouts on CPU MuJoCo and returns (N,) rewards."""
    def score(samples: np.ndarray) -> np.ndarray:
        env.snapshot_world_into_batch()
        obs = env.rollout_batch(samples, ref_wrist_pos, ref_wrist_quat, cfg.n_substeps)
        return compute_horizon_reward(
            obs=obs,
            ref_qpos=ref_qpos,
            ref_cube_pos=ref_cube_pos,
            ref_cube_quat=ref_cube_quat,
            ref_wrist_quat=ref_wrist_quat,
            contact_mask=contact_mask,
            contact_target_cube=contact_target_cube,
            contact_target_valid=contact_target_valid,
            actions=samples,
            weights=cfg.weights,
            contact_target_per_step=contact_target_per_step,
        )
    return score


def optimize_step(
    score_fn,                         # callable(samples) -> (N,) rewards
    plan: np.ndarray,                 # (H, 23)
    cfg: MPPIConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """One MPC tick: refine `plan` via `cfg.max_iters` of MPPI iterations."""
    info = {"rew_max": [], "rew_mean": []}

    for it in range(cfg.max_iters):
        noise_scale = cfg.beta_traj ** it
        samples = _sample_actions(plan, cfg, noise_scale, rng)
        rews = score_fn(samples)

        weights = _topk_softmax(rews, cfg.top_frac, cfg.softmax_temp)
        plan = (weights[:, None, None] * samples).sum(axis=0)
        # Re-clip wrist after weighted average
        plan[:, N_FINGER_CTRL:] = np.clip(
            plan[:, N_FINGER_CTRL:],
            -cfg.wrist_delta_clip,
            cfg.wrist_delta_clip,
        )
        info["rew_max"].append(float(rews.max()))
        info["rew_mean"].append(float(rews.mean()))
    return plan, info
