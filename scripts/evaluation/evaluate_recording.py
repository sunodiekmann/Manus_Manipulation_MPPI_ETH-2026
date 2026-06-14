"""Evaluate a single dynamic retargeting run.

Loads one `retargeted_sequence_dynamic[__<tag>].npz` and reports the same
metrics that `analyze_sweeps.py` computes for base runs:

  success        — completed without a drop (max pos err <= 100 mm)
  rot_mean/max   — mean / max orientation error vs reference (°)
  pos_mean/max   — mean / max position error vs reference (mm)
  C_par          — along-axis rotation coverage (~1.0 = matched reference)
  C_perp         — off-axis (wobble) coverage
  fin_geo        — geodesic orientation error at the final tick (°)
  wall_s         — total wall-clock time (s)

Also shows two plots:
  1. Per-tick orientation and position error histories.
  2. Cumulative off-axis wobble vs reference.

Usage:
    python -m scripts.evaluation.evaluate_recording recordings/<session>
    python -m scripts.evaluation.evaluate_recording recordings/<session> --tag base_s0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluation.metrics import (
    REF_DT,
    cum_perp_history_deg,
    dominant_axis,
    is_success,
    load_run,
    ref_n_for_session,
)

mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def _metric_table(run: dict, success: bool) -> None:
    ok = "YES" if success else "NO (drop)"
    print(f"\n{'─'*52}")
    print(f"  Session : {run['path'].parent.name}")
    print(f"  Tag     : {run['tag']}")
    print(f"  Ticks   : {run['n']}  ({run['n'] * REF_DT:.1f} s)")
    print(f"  Success : {ok}")
    print(f"{'─'*52}")
    print(f"  {'metric':<22}  {'value':>12}")
    print(f"  {'──────':<22}  {'─────':>12}")
    rows = [
        ("rot_mean (°)",    f"{run['rot_mean']:.2f}"),
        ("rot_max  (°)",    f"{run['rot_max']:.2f}"),
        ("pos_mean (mm)",   f"{run['pos_mean_mm']:.1f}"),
        ("pos_max  (mm)",   f"{run['pos_max_mm']:.1f}"),
        ("C_par",           f"{run['c_par']:.3f}"),
        ("C_perp",          f"{run['c_perp']:.3f}"),
        ("fin_geo  (°)",    f"{run['fin_geo']:.2f}"),
        ("wall     (s)",    f"{run['wall_s']:.0f}"),
    ]
    for name, val in rows:
        print(f"  {name:<22}  {val:>12}")
    print(f"{'─'*52}\n")


def _plot_errors(run: dict, ax_rot: plt.Axes, ax_pos: plt.Axes) -> None:
    N = run["n"]
    t = np.arange(N) * REF_DT
    ax_rot.plot(t, run["rot_history"], color="C0", lw=1.5)
    ax_rot.set_ylabel(r"$e_{\rm rot}$ (°)")
    ax_rot.set_title("Orientation error vs reference")

    ax_pos.plot(t, run["pos_history"], color="C1", lw=1.5)
    ax_pos.set_ylabel(r"$e_{\rm pos}$ (mm)")
    ax_pos.set_xlabel("time (s)")
    ax_pos.set_title("Position error vs reference")


def _plot_wobble(run: dict, ax: plt.Axes) -> None:
    N = run["n"]
    t_sim = np.arange(N) * REF_DT
    t_ref = np.arange(len(run["perp_ref_history"])) * REF_DT

    ax.plot(t_ref, run["perp_ref_history"], color="black", lw=1.8,
            ls="--", label="reference")
    ax.plot(t_sim, run["perp_sim_history"], color="C1", lw=2.0, label="sim")
    ax.set_ylabel(r"cumul. off-axis rotation $\Sigma|v_{\perp}|$ (°)")
    ax.set_xlabel("time (s)")
    ax.set_title("Cumulative off-axis wobble")
    ax.legend(fontsize=9)


def evaluate(session_dir: Path, tag: str | None) -> None:
    npz_name = (f"retargeted_sequence_dynamic__{tag}.npz" if tag
                else "retargeted_sequence_dynamic.npz")
    npz_path = session_dir / npz_name
    if not npz_path.exists():
        print(f"[error] {npz_path} not found.", file=sys.stderr)
        sys.exit(1)

    run = load_run(npz_path)
    ref_n = ref_n_for_session(session_dir)
    success = is_success(run, ref_n)

    _metric_table(run, success)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_errors(run, axes[0], axes[1])
    _plot_wobble(run, axes[2])
    fig.suptitle(f"{session_dir.name}  —  tag={run['tag']}", fontsize=11)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path,
                        help="Path to recordings/<session>")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix of the dynamic npz to load "
                             "(retargeted_sequence_dynamic__<tag>.npz). "
                             "Defaults to the canonical untagged output.")
    args = parser.parse_args()
    evaluate(args.session_dir, args.tag)


if __name__ == "__main__":
    main()
