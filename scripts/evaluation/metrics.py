"""Shared metrics helpers for Stage 3 dynamic retargeting evaluation.

Used by both `analyze_sweeps.py` (multi-session, multi-seed aggregation)
and `evaluate_recording.py` (single-session report).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REF_DT = 0.05          # ctrl_dt, seconds per tick
DROP_THRESH_M = 0.10   # pos error (m) above which a run counts as a drop


# ── rotation geometry helpers ─────────────────────────────────────────────────

def ang_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    d = abs(float(np.dot(qa, qb)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def dominant_axis(quats_wxyz: np.ndarray) -> np.ndarray:
    """Unit axis of the total rotation from frame 0 to last frame."""
    if quats_wxyz.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0])
    rots = R.from_quat(wxyz_to_xyzw(quats_wxyz[[0, -1]]))
    total = rots[1] * rots[0].inv()
    rotvec = total.as_rotvec()
    n = float(np.linalg.norm(rotvec))
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return rotvec / n


def cum_perp_history_deg(quats_wxyz: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Per-tick cumulative magnitude of rotation perpendicular to `axis`.
    Returns shape (N,) in degrees starting at 0.
    """
    if quats_wxyz.shape[0] < 2:
        return np.zeros(quats_wxyz.shape[0])
    rots = R.from_quat(wxyz_to_xyzw(quats_wxyz))
    inc = rots[1:] * rots[:-1].inv()
    vec = inc.as_rotvec()
    par_scalar = vec @ axis
    perp_vec = vec - par_scalar[:, None] * axis
    perp_mag = np.linalg.norm(perp_vec, axis=1)
    cum = np.zeros(quats_wxyz.shape[0])
    cum[1:] = np.cumsum(np.degrees(perp_mag))
    return cum


def cum_par_total_deg(quats_wxyz: np.ndarray, axis: np.ndarray) -> float:
    """Total Σ|v_par| over the trajectory, in degrees."""
    if quats_wxyz.shape[0] < 2:
        return 0.0
    rots = R.from_quat(wxyz_to_xyzw(quats_wxyz))
    inc = rots[1:] * rots[:-1].inv()
    vec = inc.as_rotvec()
    par_scalar = vec @ axis
    return float(np.degrees(np.abs(par_scalar).sum()))


# ── per-run load + metric extraction ─────────────────────────────────────────

def load_run(path: Path) -> dict:
    """Load one dynamic retargeting npz and compute all scalar metrics."""
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["run_config"])) if "run_config" in d.files else {}
    pos = np.asarray(d["pos_err_history"], dtype=float)
    rot = np.degrees(np.asarray(d["rot_err_history"], dtype=float))
    cube_sim = np.asarray(d["cube_traj"], dtype=float)
    cube_ref = np.asarray(d["cube_ref_used"], dtype=float)

    axis = dominant_axis(cube_ref[:, 3:7])
    par_sim_total  = cum_par_total_deg(cube_sim[:, 3:7], axis)
    par_ref_total  = cum_par_total_deg(cube_ref[:, 3:7], axis)
    perp_sim_hist  = cum_perp_history_deg(cube_sim[:, 3:7], axis)
    perp_ref_hist  = cum_perp_history_deg(cube_ref[:, 3:7], axis)
    perp_sim_total = float(perp_sim_hist[-1])
    perp_ref_total = float(perp_ref_hist[-1])

    c_par  = par_sim_total  / par_ref_total  if par_ref_total  > 1e-6 else float("nan")
    c_perp = perp_sim_total / perp_ref_total if perp_ref_total > 1e-6 else float("nan")

    iters_used_mean = float("nan")
    iters_max = cfg.get("iters")
    if "rew_max" in d.files:
        rew_max = np.asarray(d["rew_max"], dtype=float)
        iters_used = np.sum(~np.isnan(rew_max), axis=1)
        iters_used_mean = float(iters_used.mean())

    return {
        "path": path,
        "tag": path.stem.replace("retargeted_sequence_dynamic__", ""),
        "cfg": cfg,
        "n": int(pos.shape[0]),
        "rot_history": rot,
        "pos_history": pos * 1000.0,
        "perp_sim_history": perp_sim_hist,
        "perp_ref_history": perp_ref_hist,
        "rot_mean": float(rot.mean()),
        "rot_max": float(rot.max()),
        "pos_mean_mm": float(pos.mean() * 1000.0),
        "pos_max_mm": float(pos.max() * 1000.0),
        "c_par":  c_par,
        "c_perp": c_perp,
        "fin_geo": ang_deg(cube_sim[-1, 3:7], cube_ref[-1, 3:7]),
        "iters_used_mean": iters_used_mean,
        "iters_max_cfg":   float(iters_max) if iters_max is not None else float("nan"),
        "wall_s": cfg.get("wall_time_s", float("nan")),
        "ticks_s": cfg.get("ticks_per_s", float("nan")),
    }


def ref_n_for_session(session_dir: Path) -> int | None:
    """Expected reference length (ticks) from base runs or Stage 2 npz."""
    base = sorted(session_dir.glob("retargeted_sequence_dynamic__base_s*.npz"))
    if base:
        return max(int(np.load(p, allow_pickle=True)["cube_ref_used"].shape[0])
                   for p in base)
    s2 = session_dir / "retargeted_sequence.npz"
    if s2.exists():
        return int(np.asarray(np.load(s2)["timeline"]).shape[0])
    return None


def is_success(run: dict, ref_n: int | None,
               drop_thresh_m: float = DROP_THRESH_M) -> bool:
    completed = (ref_n is None) or (run["n"] >= ref_n)
    no_drop = run["pos_max_mm"] / 1000.0 <= drop_thresh_m
    return completed and no_drop


# ── multi-seed aggregation ────────────────────────────────────────────────────

def aggregate(paths: list[Path], ref_n: int | None) -> dict:
    runs = [load_run(p) for p in paths]
    successes = [is_success(r, ref_n) for r in runs]

    def stat(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in runs], dtype=float)
        return float(np.nanmean(v)), float(np.nanstd(v))

    return {
        "n_runs": len(runs),
        "n_success": sum(successes),
        "success_rate": sum(successes) / len(runs),
        "rot_mean": stat("rot_mean"),
        "rot_max": stat("rot_max"),
        "pos_mean": stat("pos_mean_mm"),
        "pos_max": stat("pos_max_mm"),
        "c_par":  stat("c_par"),
        "c_perp": stat("c_perp"),
        "fin_geo": stat("fin_geo"),
        "iters_used_mean": stat("iters_used_mean"),
        "iters_max_cfg":   stat("iters_max_cfg"),
        "wall_s": stat("wall_s"),
        "ticks_s": stat("ticks_s"),
        "runs": runs,
    }
