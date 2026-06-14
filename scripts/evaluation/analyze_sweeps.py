"""Thesis-grade aggregation + figure generation for Stage 3 sweeps.

Discovers tagged runs per session, groups by sweep-class, computes per-group
mean ± std across seeds, prints markdown tables, and saves figures to
`documents/figures/`.

Sweeps handled (all in `recordings/cube_pitch2/` unless otherwise noted):
  * Base runs (cube_pitch2 / cube_roll2 / cube_yaw2): `base_s<N>`.
  * MPPI N sweep (cube_pitch2):    `sweepN_N<XXXX>_s<N>`.
  * MPPI H sweep (cube_pitch2):    `sweepH_H<XX>_s<N>`.
  * MPPI iters sweep (cube_pitch2): `sweepI_I<XX>_s<N>`.
  * Edge sweep (cube_pitch2):      `edge_e<XXX>_s<N>` (XXX = mm).
  * Mass sweep (cube_pitch2):      `mass_m<XXX>_s<N>` (XXX = g).

Usage
-----
    .venv/bin/python -m scripts.evaluation.analyze_sweeps
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scripts.evaluation.metrics import (
    REF_DT, DROP_THRESH_M,
    ang_deg as _ang_deg,
    cum_perp_history_deg as _cum_perp_history_deg,
    load_run, ref_n_for_session, is_success, aggregate,
)

mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "documents" / "figures"
REC_DIR = REPO_ROOT / "recordings"
SESSIONS = ("cube_pitch2", "cube_roll2", "cube_yaw2")
MOTION_LABEL = {
    "cube_pitch2": "Cube Pitch Motion",
    "cube_roll2":  "Cube Roll Motion",
    "cube_yaw2":   "Cube Yaw Motion",
}


# ── grouping by tag regex ─────────────────────────────────────────────
def _numeric_key(s: str) -> float:
    """Pull the first integer/float out of a label like 'N=128', 'edge 50 mm'."""
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group(0)) if m else float("inf")


def discover_groups(session_dir: Path,
                    pattern: re.Pattern,
                    keyfunc) -> dict[str, list[Path]]:
    """Returns {group_key: [path, ...]}, sorted by the numeric value in the key
    (so 'N=128' comes before 'N=256' before 'N=1024')."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(session_dir.glob("retargeted_sequence_dynamic__*.npz")):
        tag = p.stem.replace("retargeted_sequence_dynamic__", "")
        m = pattern.match(tag)
        if m:
            groups[keyfunc(m)].append(p)
    return dict(sorted(groups.items(), key=lambda kv: _numeric_key(kv[0])))


# ── markdown tables ───────────────────────────────────────────────────
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def fmt_mean_std(ms: tuple[float, float], n_decimals: int = 2) -> str:
    mean, std = ms
    return f"{mean:.{n_decimals}f} ± {std:.{n_decimals}f}"


_COL_HDRS = [
    r"$n_{\text{ok}}/n$",
    r"$\bar{e}_{\text{rot}}$ (°)",
    r"$e_{\text{rot}}^{\max}$ (°)",
    r"$\bar{e}_{\text{pos}}$ (mm)",
    r"$e_{\text{pos}}^{\max}$ (mm)",
    r"$C_{\parallel}$",
    r"$C_{\perp}$",
    r"$e_{\text{geo}}^{\text{fin}}$ (°)",
    r"$T_{\text{wall}}$ (s)",
]


def _agg_cells(agg: dict) -> list[str]:
    return [
        f"{agg['n_success']}/{agg['n_runs']}",
        fmt_mean_std(agg["rot_mean"]),
        fmt_mean_std(agg["rot_max"], 1),
        fmt_mean_std(agg["pos_mean"], 1),
        fmt_mean_std(agg["pos_max"], 1),
        fmt_mean_std(agg["c_par"]),
        fmt_mean_std(agg["c_perp"]),
        fmt_mean_std(agg["fin_geo"]),
        fmt_mean_std(agg["wall_s"], 0),
    ]


def base_table(per_session: dict[str, dict]) -> str:
    rows = [[MOTION_LABEL.get(sess, sess)] + _agg_cells(agg)
            for sess, agg in per_session.items()]
    return md_table(["motion", *_COL_HDRS], rows)


def sweep_table(groups: dict, base_agg: dict | None, axis_label: str) -> str:
    rows = [[f"**{key}**"] + _agg_cells(agg) for key, agg in groups.items()]
    if base_agg is not None:
        rows.append(["*base*"] + _agg_cells(base_agg))
    return md_table([axis_label, *_COL_HDRS], rows)


# ── plots ────────────────────────────────────────────────────────────
def plot_base_per_tick(per_session: dict[str, dict], out_path: Path):
    """2×3 grid of per-tick error histories across the three captured
    motions:
        row 0 — instantaneous rotation error $e_{\text{rot}}$ (°)
        row 1 — instantaneous translation error $e_{\text{pos}}$ (mm)
    Each cell shows 5 individual seeds (light blue) plus the per-tick mean
    and ±1σ envelope (orange)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7),
                             sharex="col", sharey="row")

    rows_meta = [
        ("rot_history", r"$e_{\rm rot}$ (°)"),
        ("pos_history", r"$e_{\rm pos}$ (mm)"),
    ]

    for col, (sess, agg) in enumerate(per_session.items()):
        runs = agg["runs"]
        max_n = max(r["n"] for r in runs)
        t_full = np.arange(max_n) * REF_DT

        for row, (history_key, ylabel) in enumerate(rows_meta):
            ax = axes[row, col]
            for r in runs:
                tt = np.arange(r["n"]) * REF_DT
                ax.plot(tt, r[history_key], color="C0", alpha=0.30, lw=0.7)
            arr = np.full((len(runs), max_n), np.nan)
            for i, r in enumerate(runs):
                arr[i, :r["n"]] = r[history_key]
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            ax.fill_between(t_full, mean - std, mean + std, alpha=0.30,
                            color="C1", label=r"$\pm 1\sigma$", zorder=2)
            ax.plot(t_full, mean, color="C1", lw=2.0, label="mean", zorder=3)
            ax.set_ylabel(ylabel)
            if row == len(rows_meta) - 1:
                label = (f"{MOTION_LABEL.get(sess, sess)}  "
                         f"({agg['n_success']}/{agg['n_runs']} ok)  —  time (s)")
                ax.set_xlabel(label)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path}")
    plt.close(fig)


def iters_early_stop_table(groups: dict, base_agg: dict | None) -> str:
    """Per iters-sweep configuration: configured max iters, actual mean iters
    used (NaN-padded `rew_max` columns count as 'not used'), and the
    fraction of the budget that was actually consumed. Lets us see at a
    glance when SPIDER-style early stopping fires."""
    def row(label, agg):
        used = agg["iters_used_mean"]    # (mean, std)
        cap  = agg["iters_max_cfg"][0]   # configured max
        frac = used[0] / cap if cap > 0 else float("nan")
        return [
            label,
            f"{int(cap)}",
            f"{used[0]:.2f} ± {used[1]:.2f}",
            f"{frac*100:.0f}%",
            f"{agg['n_success']}/{agg['n_runs']}",
            fmt_mean_std(agg["rot_mean"]),
            fmt_mean_std(agg["fin_geo"]),
            fmt_mean_std(agg["wall_s"], 0),
        ]
    rows = [row(f"**{key}**", agg) for key, agg in groups.items()]
    if base_agg is not None:
        rows.append(row("*base*", base_agg))
    return md_table(
        ["sweep point",
         r"$M_{\max}$",
         r"$\langle M_{\text{used}}\rangle$",
         r"$M_{\text{used}}/M_{\max}$",
         r"$n_{\text{ok}}/n$",
         r"$\bar{e}_{\text{rot}}$ (°)",
         r"$e_{\text{geo}}^{\text{fin}}$ (°)",
         r"$T_{\text{wall}}$ (s)"],
        rows)


def opt_vs_base_table(per_motion_base: dict[str, dict],
                      per_motion_opt:  dict[str, dict]) -> str:
    """Per-motion comparison table. Columns: base → opt with % delta."""
    def delta(base_v, opt_v, lower_is_better=True):
        if base_v == 0 or np.isnan(base_v) or np.isnan(opt_v):
            return "—"
        pct = 100.0 * (opt_v - base_v) / base_v
        # Lower-is-better metrics: negative pct = good (greener).
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.0f}%"

    def cell(b_ms: tuple[float, float], o_ms: tuple[float, float],
             dec: int = 2) -> str:
        bm, bs = b_ms
        om, os = o_ms
        return (f"{bm:.{dec}f} ± {bs:.{dec}f} → "
                f"{om:.{dec}f} ± {os:.{dec}f} "
                f"({delta(bm, om)})")

    rows = []
    for sess in per_motion_base.keys():
        b = per_motion_base[sess]
        if sess not in per_motion_opt:
            continue
        o = per_motion_opt[sess]
        rows.append([
            MOTION_LABEL.get(sess, sess),
            f"{b['n_success']}/{b['n_runs']} → {o['n_success']}/{o['n_runs']}",
            cell(b["wall_s"],   o["wall_s"],   dec=0),
            cell(b["rot_mean"], o["rot_mean"], dec=2),
            cell(b["fin_geo"],  o["fin_geo"],  dec=2),
            cell(b["c_perp"],   o["c_perp"],   dec=2),
        ])
    return md_table(
        ["motion",
         r"$n_{\text{ok}}/n$ base→opt",
         r"$T_{\text{wall}}$ (s) base→opt",
         r"$\bar{e}_{\text{rot}}$ (°) base→opt",
         r"$e_{\text{geo}}^{\text{fin}}$ (°) base→opt",
         r"$C_{\perp}$ base→opt"],
        rows)


def plot_opt_vs_base(per_motion_base: dict[str, dict],
                     per_motion_opt:  dict[str, dict],
                     out_path: Path):
    """1×3 grouped bar chart: wall_s | rot_mean | fin_geo, base vs opt,
    grouped by motion. Δ% annotated above each opt bar."""
    motions = [m for m in per_motion_base if m in per_motion_opt]
    metrics = [
        ("wall_s",   "wall-clock (s)",          0),
        ("rot_mean", r"$\bar{e}_{\rm rot}$ (°)", 2),
        ("fin_geo",  r"$e_{\rm geo}^{\rm fin}$ (°)", 2),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.8))

    x = np.arange(len(motions))
    width = 0.35
    for ax, (key, ylabel, dec) in zip(axes, metrics):
        base_vals  = [per_motion_base[m][key][0] for m in motions]
        base_errs  = [per_motion_base[m][key][1] for m in motions]
        opt_vals   = [per_motion_opt[m][key][0]  for m in motions]
        opt_errs   = [per_motion_opt[m][key][1]  for m in motions]

        ax.bar(x - width/2, base_vals, width, yerr=base_errs, capsize=4,
               color="C0", label="base", edgecolor="black", lw=0.5)
        ax.bar(x + width/2, opt_vals, width, yerr=opt_errs, capsize=4,
               color="C1", label="opt", edgecolor="black", lw=0.5)

        # % delta annotations above the opt bars
        for i, (b, o) in enumerate(zip(base_vals, opt_vals)):
            if b == 0 or np.isnan(b) or np.isnan(o):
                continue
            pct = 100.0 * (o - b) / b
            color = "green" if pct < 0 else "firebrick"
            ax.annotate(f"{'+' if pct >= 0 else ''}{pct:.0f}%",
                        xy=(i + width/2, o + max(opt_errs[i], 0)),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, color=color,
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MOTION_LABEL.get(m, m).replace(" Motion", "")
                            for m in motions])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_wobble_per_tick(per_session: dict[str, dict], out_path: Path):
    """1×3 grid: cumulative perpendicular rotation (off-axis wobble) over
    time, one panel per motion.

    Reference is plotted as dashed black (the small, steady off-axis
    content of the captured motion). Sim is per-seed light blue + mean ± σ
    in orange. When the sim curves grow faster than the reference, the
    planner is adding incidental rotation around axes the human didn't."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for col, (sess, agg) in enumerate(per_session.items()):
        runs = agg["runs"]
        max_n = max(r["n"] for r in runs)
        t_full = np.arange(max_n) * REF_DT
        ax = axes[col]

        # reference: same across seeds — pick the longest
        longest_ref = max((r["perp_ref_history"] for r in runs), key=len)
        t_ref = np.arange(len(longest_ref)) * REF_DT
        ax.plot(t_ref, longest_ref, color="black", lw=1.8, ls="--",
                label="reference", zorder=4)

        # sim per-seed
        for r in runs:
            tt = np.arange(r["n"]) * REF_DT
            ax.plot(tt, r["perp_sim_history"], color="C0", alpha=0.30, lw=0.7)

        # mean ± std
        arr = np.full((len(runs), max_n), np.nan)
        for i, r in enumerate(runs):
            arr[i, :r["n"]] = r["perp_sim_history"]
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        ax.fill_between(t_full, mean - std, mean + std, alpha=0.30,
                        color="C1", zorder=2)
        ax.plot(t_full, mean, color="C1", lw=2.0, label="sim mean", zorder=3)

        ax.set_xlabel(MOTION_LABEL.get(sess, sess))
        if col == 0:
            ax.set_ylabel(r"cumulative off-axis rotation $\Sigma\,|v_{\perp}|$ (°)")
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_mppi_pareto(n_groups: dict, h_groups: dict, i_groups: dict,
                     base_agg: dict, out_path: Path,
                     base_labels: tuple[str, str, str] = ("N=1024", "H=16", "I=12")):
    """2×3 grid:
        rows = metric (rot_mean top / fin_geo bottom)
        cols = swept axis (samples / horizon / iters)
       x-axis = wall-clock time, base marked with ★."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             sharex="col", sharey="row")

    axis_data = [
        (n_groups, base_labels[0], "samples (N)",   "C0", "o"),
        (h_groups, base_labels[1], "horizon (H)",   "C1", "s"),
        (i_groups, base_labels[2], "iters (I)",     "C2", "^"),
    ]

    def draw_one(ax, groups, base_label, color, marker, metric_key):
        items = list(groups.items())
        items.append((base_label, base_agg))
        items.sort(key=lambda kv: _numeric_key(kv[0]))
        xs   = [v["wall_s"][0] for _, v in items]
        xerr = [v["wall_s"][1] for _, v in items]
        ys   = [v[metric_key][0] for _, v in items]
        yerr = [v[metric_key][1] for _, v in items]

        ax.errorbar(xs, ys, yerr=yerr, xerr=xerr, color=color, marker=marker,
                    lw=1.5, capsize=3, markersize=8, zorder=3)
        for (label, _), x, y in zip(items, xs, ys):
            num = re.search(r"\d+", label)
            txt = num.group(0) if num else label
            ax.annotate(txt, (x, y), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=9, color=color)
        bx = base_agg["wall_s"][0]
        by = base_agg[metric_key][0]
        ax.plot(bx, by, marker="s", markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=2.0,
                markersize=12, zorder=5, linestyle="none", label="base")
        ax.grid(True, alpha=0.3)

    for col, (groups, base_label, title, color, marker) in enumerate(axis_data):
        draw_one(axes[0, col], groups, base_label, color, marker, "rot_mean")
        draw_one(axes[1, col], groups, base_label, color, marker, "fin_geo")
        axes[1, col].set_xlabel(f"wall-clock (s)  —  {title}")

    axes[0, 0].set_ylabel("mean rotation error (°)")
    axes[1, 0].set_ylabel("endpoint geodesic error (°)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path}")
    plt.close(fig)


def plot_physics_bubble(edge_groups: dict, mass_groups: dict,
                        base_agg: dict, out_path: Path):
    """2×2 grid:
        rows = metric (rot_mean top / fin_geo bottom)
        cols = physics axis (edge / mass)
       Bubble size + opacity encode success rate; base highlighted in red.
       No legend — caption in the markdown explains the encoding."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")

    def draw(ax, groups, value_fn, base_value, base_agg, metric_key,
             axis_label, unit):
        items = sorted(groups.items(), key=lambda kv: value_fn(kv[0]))
        xs = [value_fn(k) for k, _ in items]
        ys = [v[metric_key][0] for _, v in items]
        yerr = [v[metric_key][1] for _, v in items]
        succ_frac = [v["success_rate"] for _, v in items]
        succ_counts = [(v["n_success"], v["n_runs"]) for _, v in items]
        sizes = [80 + 1200 * s for s in succ_frac]
        colors = [(0.20, 0.40, 0.80, 0.20 + 0.75 * s) for s in succ_frac]

        ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="grey",
                    capsize=4, lw=1.0, zorder=2)
        ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="C0",
                   linewidths=1.5, zorder=3)
        for x, y, (ns, nt) in zip(xs, ys, succ_counts):
            ax.annotate(f"{ns}/{nt}", (x, y), xytext=(0, -22),
                        textcoords="offset points", fontsize=9,
                        ha="center", color="black")

        if base_agg is not None and base_value is not None:
            bx = base_value
            by = base_agg[metric_key][0]
            byerr = base_agg[metric_key][1]
            bsize = 80 + 1200 * base_agg["success_rate"]
            ax.errorbar([bx], [by], yerr=[byerr], fmt="none",
                        ecolor="grey", capsize=4, lw=1.0, zorder=2)
            ax.scatter([bx], [by], s=bsize,
                       c=[(0.85, 0.15, 0.15,
                           0.20 + 0.75 * base_agg["success_rate"])],
                       edgecolors="C3", linewidths=2.0, zorder=4)
            ax.annotate(f"{base_agg['n_success']}/{base_agg['n_runs']}",
                        (bx, by), xytext=(0, -22),
                        textcoords="offset points", fontsize=9,
                        ha="center", color="C3", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # ── rot_mean row ─────────────────────────────────────────────
    draw(axes[0, 0], edge_groups, lambda k: int(k.split()[0]),
         base_value=70, base_agg=base_agg, metric_key="rot_mean",
         axis_label="cube edge length", unit="mm")
    draw(axes[0, 1], mass_groups, lambda k: int(k.split()[0]),
         base_value=40, base_agg=base_agg, metric_key="rot_mean",
         axis_label="cube mass", unit="g")

    # ── fin_geo row ──────────────────────────────────────────────
    draw(axes[1, 0], edge_groups, lambda k: int(k.split()[0]),
         base_value=70, base_agg=base_agg, metric_key="fin_geo",
         axis_label="cube edge length", unit="mm")
    draw(axes[1, 1], mass_groups, lambda k: int(k.split()[0]),
         base_value=40, base_agg=base_agg, metric_key="fin_geo",
         axis_label="cube mass", unit="g")

    axes[0, 0].set_ylabel("mean rotation error (°)")
    axes[1, 0].set_ylabel("endpoint geodesic error (°)")
    axes[1, 0].set_xlabel("cube edge length (mm)")
    axes[1, 1].set_xlabel("cube mass (g)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  saved {out_path}")
    plt.close(fig)


# ── main flow ────────────────────────────────────────────────────────
MD_OUT = REPO_ROOT / "documents" / "sweep_analysis.md"


class _Tee:
    """Write to both stdout and an in-memory list; flush to file at the end."""
    def __init__(self):
        self.lines: list[str] = []
    def __call__(self, *args):
        s = " ".join(str(a) for a in args)
        print(s)
        self.lines.append(s)
    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n")
        print(f"\n# saved markdown → {path.relative_to(REPO_ROOT)}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = _Tee()
    out(f"# Sweep analysis\n")
    out(f"_Generated by `scripts/evaluation/analyze_sweeps.py`._  \n"
        f"_Figures saved to `{FIG_DIR.relative_to(REPO_ROOT)}/`._\n")
    out("## Metric glossary\n")
    out(r"- $n_{\text{ok}}/n$ — success rate. A run counts as success only if "
        r"it completed the full reference length AND $e_{\text{pos}}^{\max} "
        r"< 100\,$mm. Early-stopped runs (frac_term$=1$) count as failures.")
    out(r"- $\bar{e}_{\text{rot}}$, $e_{\text{rot}}^{\max}$ — per-tick cube "
        r"orientation error vs reference, mean and max over the trajectory.")
    out(r"- $\bar{e}_{\text{pos}}$, $e_{\text{pos}}^{\max}$ — same for "
        r"translation.")
    out(r"- $C_{\parallel}$ — *along-axis* rotation coverage. Per-tick "
        r"rotation increments are projected onto the dominant axis of the "
        r"reference's total rotation; $C_{\parallel}$ is the ratio "
        r"$\sum_t \lvert v_{\parallel}^{\text{sim},t}\rvert / "
        r"\sum_t \lvert v_{\parallel}^{\text{ref},t}\rvert$. $\sim 1$ means "
        r"the cube executed the captured rotation; $>1$ over-rotated along "
        r"the axis; $<1$ under-rotated.")
    out(r"- $C_{\perp}$ — *perpendicular* (off-axis) rotation coverage. "
        r"Same construction but using the perpendicular component of each "
        r"increment, $\lvert v_{\perp}\rvert = \lVert v - (v\cdot\hat a)\hat a\rVert$. "
        r"$\sim 1$ means the sim adds about as much off-axis wobble as the "
        r"reference does; $\gg 1$ means the planner introduces incidental "
        r"rotation around axes the human didn't.")
    out(r"- $e_{\text{geo}}^{\text{fin}}$ — geodesic orientation error at the "
        r"final tick (endpoint pose accuracy).")
    out(r"- $T_{\text{wall}}$ — total run time, seconds.")
    out("- All ± values are mean ± std over 5 seeds.")
    out("")

    # ─────────────────────────────────────────
    # Base runs across the three motions
    # ─────────────────────────────────────────
    out("## Base runs (5 seeds × 3 motions)\n")
    per_session_base = {}
    base_pattern = re.compile(r"base_s(\d+)$")
    for sess in SESSIONS:
        d = REC_DIR / sess
        ref_n = ref_n_for_session(d)
        paths = sorted(d.glob("retargeted_sequence_dynamic__base_s*.npz"))
        if not paths:
            continue
        per_session_base[sess] = aggregate(paths, ref_n)
    out(base_table(per_session_base))
    out("")
    out("Per-tick `rot_err` and `pos_err` over time (individual seeds in blue, "
        "mean ± std in orange):\n")
    out(f"![base per-tick](figures/base_per_tick.png)\n")
    plot_base_per_tick(per_session_base, FIG_DIR / "base_per_tick.png")
    out("")
    out("Off-axis wobble: cumulative magnitude of rotation perpendicular to "
        "each motion's dominant axis. Reference (dashed black) accumulates "
        "slowly; sim (orange) grows faster — the gap is the incidental "
        "wobble the planner introduces.\n")
    out(f"![wobble per-tick](figures/wobble_per_tick.png)\n")
    plot_wobble_per_tick(per_session_base, FIG_DIR / "wobble_per_tick.png")
    out("")

    # ─────────────────────────────────────────
    # Optimal-config comparison: base vs opt across the three motions
    # ─────────────────────────────────────────
    per_session_opt: dict[str, dict] = {}
    for sess in SESSIONS:
        d = REC_DIR / sess
        ref_n = ref_n_for_session(d)
        paths = sorted(d.glob("retargeted_sequence_dynamic__opt_s*.npz"))
        if paths:
            per_session_opt[sess] = aggregate(paths, ref_n)
    if per_session_opt:
        out("## Optimal config vs base (5 seeds × 3 motions)\n")
        out("Base config: $N=1024$, $H=16$, iters$=12$. Optimal config "
            "(from the MPPI Pareto on cube_pitch2): $N=512$, $H=20$, "
            "iters$=8$. Reward weights, physics, all other settings "
            "identical between the two runs.\n")
        out(opt_vs_base_table(per_session_base, per_session_opt))
        out("")
        out("Side-by-side bars per motion. Green = improvement, "
            "red = regression vs base:\n")
        out(f"![opt vs base](figures/opt_vs_base.png)\n")
        plot_opt_vs_base(per_session_base, per_session_opt,
                         FIG_DIR / "opt_vs_base.png")
        out("")

    # ─────────────────────────────────────────
    # MPPI sweep on cube_pitch2
    # ─────────────────────────────────────────
    pitch_dir = REC_DIR / "cube_pitch2"
    pitch_ref_n = ref_n_for_session(pitch_dir)
    base_pitch_agg = per_session_base.get("cube_pitch2")

    n_pat = re.compile(r"sweepN_N(\d+)_s(\d+)$")
    h_pat = re.compile(r"sweepH_H(\d+)_s(\d+)$")
    i_pat = re.compile(r"sweepI_I(\d+)_s(\d+)$")
    n_groups_paths = discover_groups(pitch_dir, n_pat, lambda m: f"N={int(m.group(1))}")
    h_groups_paths = discover_groups(pitch_dir, h_pat, lambda m: f"H={int(m.group(1))}")
    i_groups_paths = discover_groups(pitch_dir, i_pat, lambda m: f"I={int(m.group(1))}")

    n_groups = {k: aggregate(v, pitch_ref_n) for k, v in n_groups_paths.items()}
    h_groups = {k: aggregate(v, pitch_ref_n) for k, v in h_groups_paths.items()}
    i_groups = {k: aggregate(v, pitch_ref_n) for k, v in i_groups_paths.items()}

    if n_groups or h_groups or i_groups:
        out("## MPPI sweep — cube_pitch2 (5 seeds per config)\n")
        if n_groups:
            out(f"### N (samples) sweep — H=16, iters=12 fixed\n")
            out(sweep_table(n_groups, base_pitch_agg, "N (samples)"))
            out("")
        if h_groups:
            out(f"### H (horizon) sweep — N=1024, iters=12 fixed\n")
            out(sweep_table(h_groups, base_pitch_agg, "H (horizon steps)"))
            out("")
        if i_groups:
            out(f"### iters sweep — N=1024, H=16 fixed\n")
            out(sweep_table(i_groups, base_pitch_agg, "iters"))
            out("")
            out(f"#### Early-stopping usage per iters configuration\n")
            out("Configured max iters vs how many MPPI actually consumed on "
                "average per tick. Where actual usage is well below the cap, "
                "the SPIDER-style improvement-threshold early stop is firing "
                "(rew_max is NaN-padded for unused iterations).\n")
            out(iters_early_stop_table(i_groups, base_pitch_agg))
            out("")
        if base_pitch_agg is not None:
            out("Compute vs tracking — one panel per swept axis, base marked ★:\n")
            out(f"![mppi pareto](figures/mppi_pareto.png)\n")
            plot_mppi_pareto(n_groups, h_groups, i_groups, base_pitch_agg,
                             FIG_DIR / "mppi_pareto.png")
        out("")

    # ─────────────────────────────────────────
    # Physics: edge sweep on cube_pitch2
    # ─────────────────────────────────────────
    edge_pat = re.compile(r"edge_e(\d+)_s(\d+)$")
    edge_groups_paths = discover_groups(
        pitch_dir, edge_pat, lambda m: f"{int(m.group(1))} mm")
    edge_groups = {k: aggregate(v, pitch_ref_n)
                   for k, v in edge_groups_paths.items()}
    if edge_groups:
        out(f"## Physics sweep — cube edge length ({MOTION_LABEL['cube_pitch2']}, 5 seeds)\n")
        out("Base cube edge is 70 mm (XML default).\n")
        out(sweep_table(edge_groups, base_pitch_agg, "edge"))
        out("")

    # ─────────────────────────────────────────
    # Physics: mass sweep on cube_pitch2
    # ─────────────────────────────────────────
    mass_pat = re.compile(r"mass_m(\d+)_s(\d+)$")
    mass_groups_paths = discover_groups(
        pitch_dir, mass_pat, lambda m: f"{int(m.group(1))} g")
    mass_groups = {k: aggregate(v, pitch_ref_n)
                   for k, v in mass_groups_paths.items()}
    if mass_groups:
        out(f"## Physics sweep — cube mass ({MOTION_LABEL['cube_pitch2']}, 5 seeds)\n")
        out("Base cube mass is 40 g (XML default).\n")
        out(sweep_table(mass_groups, base_pitch_agg, "mass"))
        out("")

    out.write(MD_OUT)


if __name__ == "__main__":
    main()
