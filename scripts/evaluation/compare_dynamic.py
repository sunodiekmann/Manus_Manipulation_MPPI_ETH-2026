"""Compare Stage 3 dynamic-retargeting runs (sweep analysis).

Loads one or more `retargeted_sequence_dynamic[__<tag>].npz` files from a
session and prints a reward-agnostic metric table, so pose-tracking and
task-reward (cube_angvel) runs are compared on the same neutral yardstick.

All metrics are derived from keys the optimizer already saves
(`pos_err_history`, `rot_err_history`, `cube_traj`, `cube_ref_used`,
`ctrl_traj`, `term_*`, `run_config`). No re-simulation.

Metrics
-------
  success      : completed without a drop (max pos err <= --drop-threshold)
  rot_cover    : achieved cumulative rotation / reference cumulative rotation
                 (guards against "stay still scores low rot_err"). ~1.0 = the
                 cube turned as much as the reference; <1 = under-rotated.
  rot mean/max/fin : world-frame orientation error vs ref (deg)
  pos mean/max/fin : world-frame position error vs ref (mm)
  fin_geo      : geodesic orientation error at the LAST tick (deg)
  contact      : mean of the contact reward term (more negative = looser grip)
  d_ctrl       : mean per-tick finger-ctrl change (effort / jerk proxy)
  wdelta       : mean wrist axis-angle delta magnitude (rad)
  improv       : mean MPPI per-tick improvement
  t/s          : ticks per second (compute cost, from run_config)

Usage
-----
    # all runs in a session
    python -m scripts.evaluation.compare_dynamic recordings/cube_pitch2
    # specific tagged runs
    python -m scripts.evaluation.compare_dynamic recordings/cube_pitch2 \
        --tags baseline contact3 angvel1
    # across sessions (one file each, by explicit paths)
    python -m scripts.evaluation.compare_dynamic --files \
        recordings/cube_roll2/retargeted_sequence_dynamic.npz \
        recordings/cube_pitch2/retargeted_sequence_dynamic.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _quat_dot_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two wxyz quaternions."""
    d = abs(float(np.dot(q_a, q_b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def _cumulative_rotation_deg(quats: np.ndarray) -> float:
    """Sum of per-tick geodesic rotation increments over a (N,4) wxyz seq."""
    if quats.shape[0] < 2:
        return 0.0
    total = 0.0
    for t in range(quats.shape[0] - 1):
        total += _quat_dot_angle_deg(quats[t], quats[t + 1])
    return total


def _load_run(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [skip] missing {path}")
        return None
    d = np.load(path, allow_pickle=True)
    cfg = {}
    if "run_config" in d.files:
        try:
            cfg = json.loads(str(d["run_config"]))
        except Exception:
            cfg = {}
    return {"path": path, "npz": d, "cfg": cfg}


def _metrics(run: dict, drop_threshold_m: float) -> dict:
    d = run["npz"]
    cfg = run["cfg"]
    pos = np.asarray(d["pos_err_history"], dtype=float)        # m
    rot = np.degrees(np.asarray(d["rot_err_history"], dtype=float))  # deg
    cube_sim = np.asarray(d["cube_traj"], dtype=float)         # (N,7)
    cube_ref = (np.asarray(d["cube_ref_used"], dtype=float)
                if "cube_ref_used" in d.files else None)

    cum_sim = _cumulative_rotation_deg(cube_sim[:, 3:7])
    if cube_ref is not None:
        cum_ref = _cumulative_rotation_deg(cube_ref[:, 3:7])
        rot_cover = (cum_sim / cum_ref) if cum_ref > 1e-6 else float("nan")
        fin_geo = _quat_dot_angle_deg(cube_sim[-1, 3:7], cube_ref[-1, 3:7])
    else:
        rot_cover, fin_geo = float("nan"), float("nan")

    # contact term mean (term order from saved term_names)
    contact_mean = float("nan")
    if "term_names" in d.files and "term_means_history" in d.files:
        names = [str(n) for n in d["term_names"]]
        tmh = np.asarray(d["term_means_history"], dtype=float)
        if "contact" in names:
            contact_mean = float(tmh[:, names.index("contact")].mean())

    ctrl = np.asarray(d["ctrl_traj"], dtype=float)             # (N,20)
    d_ctrl = (float(np.linalg.norm(np.diff(ctrl, axis=0), axis=1).mean())
              if ctrl.shape[0] > 1 else 0.0)
    wdelta = float(np.linalg.norm(
        np.asarray(d["wrist_delta_traj"], dtype=float), axis=1).mean())
    improv = float(np.asarray(d["iter_improvement_history"], dtype=float).mean())

    return {
        "n": pos.shape[0],
        "success": bool(pos.max() <= drop_threshold_m),
        "rot_cover": rot_cover,
        "rot_mean": float(rot.mean()), "rot_max": float(rot.max()),
        "rot_fin": float(rot[-1]),
        "pos_mean": float(pos.mean() * 1000), "pos_max": float(pos.max() * 1000),
        "pos_fin": float(pos[-1] * 1000),
        "fin_geo": fin_geo,
        "contact": contact_mean,
        "d_ctrl": d_ctrl, "wdelta": wdelta, "improv": improv,
        "t_per_s": cfg.get("ticks_per_s", float("nan")),
    }


def _label(run: dict) -> str:
    tag = run["cfg"].get("tag")
    if tag:
        return tag
    name = run["path"].stem
    return name.replace("retargeted_sequence_dynamic", "").lstrip("_") or "canonical"


def _weight_summary(cfg: dict) -> str:
    w = cfg.get("weights", {})
    if not w:
        return ""
    keys = ["cube_pos", "cube_quat", "finger_qpos", "contact",
            "wrist_delta", "qvel", "cube_angvel"]
    return " ".join(f"{k.split('_')[-1] if k!='cube_pos' else 'pos'}="
                    f"{w.get(k, 0):g}" for k in keys)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session_dir", nargs="?", type=Path,
                   help="session dir; loads its dynamic npz(s)")
    p.add_argument("--tags", nargs="*", default=None,
                   help="specific tags to load (default: all in the session)")
    p.add_argument("--files", nargs="*", type=Path, default=None,
                   help="explicit npz paths (overrides session_dir/--tags)")
    p.add_argument("--drop-threshold", type=float, default=0.10,
                   help="max pos err (m) above which a run counts as a drop")
    args = p.parse_args()

    paths: list[Path] = []
    if args.files:
        paths = list(args.files)
    elif args.session_dir:
        if args.tags:
            for t in args.tags:
                nm = ("retargeted_sequence_dynamic.npz" if t in ("canonical", "")
                      else f"retargeted_sequence_dynamic__{t}.npz")
                paths.append(args.session_dir / nm)
        else:
            paths = sorted(args.session_dir.glob("retargeted_sequence_dynamic*.npz"))
    if not paths:
        p.error("no runs found — pass a session_dir or --files")

    runs = [r for r in (_load_run(pp) for pp in paths) if r is not None]
    if not runs:
        return
    rows = [(_label(r), _metrics(r, args.drop_threshold), r["cfg"]) for r in runs]

    cols = [
        ("run", "{:<14}", "{:<14}"),
        ("ok", "{:>3}", "{:>3}"),
        ("n", "{:>4}", "{:>4}"),
        ("rot_cov", "{:>7}", "{:>7.2f}"),
        ("rotμ°", "{:>6}", "{:>6.2f}"),
        ("rotmax", "{:>6}", "{:>6.1f}"),
        ("rotfin", "{:>6}", "{:>6.2f}"),
        ("posμmm", "{:>6}", "{:>6.1f}"),
        ("posmax", "{:>6}", "{:>6.1f}"),
        ("fingeo", "{:>6}", "{:>6.2f}"),
        ("contact", "{:>7}", "{:>7.3f}"),
        ("dctrl", "{:>6}", "{:>6.3f}"),
        ("wdelta", "{:>6}", "{:>6.3f}"),
        ("improv", "{:>6}", "{:>6.2f}"),
        ("t/s", "{:>5}", "{:>5.2f}"),
    ]
    header = "  ".join(h.format(name) for name, h, _ in cols)
    print(header)
    print("-" * len(header))
    keymap = {
        "rot_cov": "rot_cover", "rotμ°": "rot_mean", "rotmax": "rot_max",
        "rotfin": "rot_fin", "posμmm": "pos_mean", "posmax": "pos_max",
        "fingeo": "fin_geo", "contact": "contact", "dctrl": "d_ctrl",
        "wdelta": "wdelta", "improv": "improv", "t/s": "t_per_s",
    }
    for label, m, _cfg in rows:
        cells = []
        for name, _hfmt, vfmt in cols:
            if name == "run":
                cells.append(vfmt.format(label[:14]))
            elif name == "ok":
                cells.append(vfmt.format("Y" if m["success"] else "N"))
            elif name == "n":
                cells.append(vfmt.format(m["n"]))
            else:
                v = m[keymap[name]]
                cells.append(vfmt.format(v) if v == v else f"{'--':>{len(vfmt.format(0))}}")
        print("  ".join(cells))

    print("\nconfigs:")
    for label, _m, cfg in rows:
        ws = _weight_summary(cfg)
        meta = (f"N={cfg.get('num_samples','?')} it={cfg.get('iters','?')} "
                f"H={cfg.get('horizon','?')}") if cfg else "(no run_config)"
        print(f"  {label:<14} {meta}  {ws}")


if __name__ == "__main__":
    main()
