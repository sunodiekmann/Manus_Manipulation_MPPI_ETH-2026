"""Visualize per-finger contact schedule over time.

For each of the 5 fingers, plots a stack of bands sharing a common x-axis:

  - 2a raw face   : Stage 2a (cube_relative IK) tip's nearest-face id, per
                    frame. "If this finger touched something now, which face
                    would it touch?" Doesn't say whether contact is actually
                    happening; just the dominant cube-frame axis.
  - scheduled face: spider_contact's per-frame segmented contact target,
                    colored. Grey where the finger is not flagged as in
                    contact (NaN in contact_target_per_frame).
  - mask          : black where spider_contact's contact_mask is True (tip
                    geom collided with cube geom in mj_forward). This gates
                    whether the finger appears in the schedule at all.
  - 2c face       : same FK-based face id but applied to the Stage 2c
                    (retargeted_sequence_feasible) qpos. Shows what the
                    feasibility IK actually drives the finger to, including
                    the smooth snap pulling tips onto the surface.
  - SD (mm)       : signed distance of Stage 2a tip to nearest cube face,
                    drawn as a thin grey trace. 0 = on surface (dashed
                    reference line), positive = outside, negative = inside.

Usage:
    python -m scripts.visualization.plot_contact_schedule recordings/<session>
    python -m scripts.visualization.plot_contact_schedule recordings/<session> --show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from scripts.retargeting.manual_contacts import (
    FORBIDDEN, UNCONSTRAINED, load_constraints,
)
from scripts.retargeting.spider_contact import _cube_face_id
from scripts.utils.config import CUBE_SIZE
from scripts.utils.transforms import pose_to_matrix

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]

# Stable, distinct colors per cube face. Pairs of related hues for ±axis.
FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]
FACE_COLORS = [
    "#d73027",   # +x  red
    "#fdae61",   # -x  light orange
    "#1a9850",   # +y  green
    "#a6d96a",   # -y  light green
    "#4575b4",   # +z  blue
    "#abd9e9",   # -z  light blue
]
NO_CONTACT_COLOR = "#ededed"
CUBE_HALF = CUBE_SIZE / 2


def _compute_raw_face_and_sd(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    cube_pose_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each (frame, finger), compute face id (0-5) and signed distance
    of the FK fingertip to the nearest cube face, in cube frame. NaN where
    qpos or cube pose is missing.
    """
    data = mujoco.MjData(model)
    tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in TIP_GEOMS]
    if any(g < 0 for g in tip_gids):
        missing = [n for n, g in zip(TIP_GEOMS, tip_gids) if g < 0]
        raise ValueError(f"missing tip geoms: {missing}")
    cube_qadr_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qadr = int(model.jnt_qposadr[cube_qadr_jid])

    N = qpos_seq.shape[0]
    face_out = np.full((N, 5), -1, dtype=int)
    sd_out = np.full((N, 5), np.nan)
    for t in range(N):
        q = qpos_seq[t]
        cp = cube_pose_seq[t]
        if np.any(np.isnan(q)) or np.any(np.isnan(cp)):
            continue
        data.qpos[:27] = q
        data.qpos[cube_qadr : cube_qadr + 7] = cp
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        T = pose_to_matrix(cp)
        R, t_wc = T[:3, :3], T[:3, 3]
        for f, gid in enumerate(tip_gids):
            tip_world = data.geom_xpos[gid]
            tip_cube = R.T @ (tip_world - t_wc)
            face_out[t, f] = _cube_face_id(tip_cube)
            sd_out[t, f] = float(np.abs(tip_cube).max() - CUBE_HALF)
    return face_out, sd_out


def _cleaned_face_per_frame(target_per_frame: np.ndarray) -> np.ndarray:
    """(N, 5, 3) cube-frame segmented targets → (N, 5) face ids, -1 where NaN."""
    N, F = target_per_frame.shape[:2]
    out = np.full((N, F), -1, dtype=int)
    for t in range(N):
        for f in range(F):
            p = target_per_frame[t, f]
            if not np.any(np.isnan(p)):
                out[t, f] = _cube_face_id(p)
    return out


def _draw_face_band(ax, ts: np.ndarray, face_seq: np.ndarray, y_lo: float, y_hi: float):
    """Draw a horizontal colored band: a rect per contiguous same-face run.
    -1 face id → no-contact grey. Bands span [y_lo, y_hi] vertically.
    """
    N = len(face_seq)
    if N == 0:
        return
    dt = float(np.median(np.diff(ts))) if N > 1 else 1.0
    # Walk and build runs.
    start = 0
    cur = int(face_seq[0])
    for t in range(1, N + 1):
        if t == N or int(face_seq[t]) != cur:
            x_lo = float(ts[start]) - dt * 0.5
            x_hi = float(ts[t - 1]) + dt * 0.5
            color = FACE_COLORS[cur] if cur >= 0 else NO_CONTACT_COLOR
            ax.add_patch(mpatches.Rectangle(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                facecolor=color, edgecolor="none",
            ))
            if t < N:
                start = t
                cur = int(face_seq[t])


def plot_session(session_dir: Path, show: bool = False):
    cs_path = session_dir / "contact_sequence.npz"
    rel_path = session_dir / "retargeted_sequence_cube_relative.npz"
    if not cs_path.exists():
        raise FileNotFoundError(cs_path)
    if not rel_path.exists():
        raise FileNotFoundError(rel_path)

    cs = np.load(cs_path)
    rel = np.load(rel_path)
    timeline = np.asarray(cs["timeline"], dtype=float)
    target_pf = np.asarray(cs["contact_target_per_frame"], dtype=float)
    contact_mask = np.asarray(cs["contact_mask"], dtype=bool)

    cleaned = _cleaned_face_per_frame(target_pf)
    manual = load_constraints(session_dir, timeline)
    if manual is not None:
        print(f"  loaded manual constraints: "
              f"{int((manual == FORBIDDEN).sum())} forbid, "
              f"{int((manual >= 0).sum())} face-bound")

    print(f"  computing FK on Stage 2a qpos to get raw face + SD ({len(timeline)} frames) …")
    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    raw_face, raw_sd = _compute_raw_face_and_sd(
        model,
        np.asarray(rel["retargeted_qpos"], dtype=float),
        np.asarray(rel["cube_pose_used"], dtype=float),
    )

    # Stage 2c face id (optional — if feasibility output exists).
    feas_path = session_dir / "retargeted_sequence_feasible.npz"
    if feas_path.exists():
        feas = np.load(feas_path)
        print(f"  computing FK on Stage 2c qpos …")
        feas_face, feas_sd = _compute_raw_face_and_sd(
            model,
            np.asarray(feas["retargeted_qpos"], dtype=float),
            np.asarray(feas["cube_pose_smoothed"], dtype=float),
        )
    else:
        print(f"  [info] no {feas_path.name}; 2c band will be empty")
        feas_face = np.full_like(raw_face, -1)
        feas_sd = np.full_like(raw_sd, np.nan)

    ts = timeline - timeline[0]  # seconds from start
    N = len(ts)

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    # Band y-coordinates (stacked top-to-bottom in each finger subplot).
    BAND_MANUAL = (4.0, 4.3)     # manual-constraint indicator (top, if present)
    BAND_2A = (3.2, 3.8)        # Stage 2a raw face id
    BAND_SCH = (2.4, 3.0)       # scheduled face id
    BAND_MASK = (2.05, 2.20)     # mask strip
    BAND_2C = (1.2, 1.8)         # Stage 2c face id
    SD_TOP = 0.95                # SD line band top
    SD_BOTTOM = -0.6             # SD line band bottom (negative SD area)
    SD_SCALE_MIN, SD_SCALE_MAX = -30.0, 30.0  # mm shown
    dt_med = float(np.median(np.diff(ts))) if N > 1 else 1.0
    have_manual = manual is not None

    def _sd_to_norm(mm):
        # Higher SD (further from cube) → lower y on the plot.
        # SD=0 maps to upper part of band; large positive SD goes lower.
        frac = (np.clip(mm, SD_SCALE_MIN, SD_SCALE_MAX) - SD_SCALE_MIN) / (SD_SCALE_MAX - SD_SCALE_MIN)
        return SD_TOP - frac * (SD_TOP - SD_BOTTOM)

    for f, (ax, fn) in enumerate(zip(axes, FINGER_NAMES)):
        # Bands (top to bottom).
        if have_manual:
            # Manual: face id for face-bound frames (colored), grey for FORBIDDEN,
            # invisible (no draw) for UNCONSTRAINED so the band breathes.
            for t in range(N):
                code = int(manual[t, f])
                if code == UNCONSTRAINED:
                    continue
                if code == FORBIDDEN:
                    color = "#000000"  # black for forbid
                else:
                    color = FACE_COLORS[code]
                ax.add_patch(mpatches.Rectangle(
                    (ts[t] - dt_med * 0.5, BAND_MANUAL[0]), dt_med,
                    BAND_MANUAL[1] - BAND_MANUAL[0],
                    facecolor=color, edgecolor="none",
                ))
        _draw_face_band(ax, ts, raw_face[:, f],   y_lo=BAND_2A[0],  y_hi=BAND_2A[1])
        _draw_face_band(ax, ts, cleaned[:, f],    y_lo=BAND_SCH[0], y_hi=BAND_SCH[1])
        for t in range(N):
            if contact_mask[t, f]:
                ax.add_patch(mpatches.Rectangle(
                    (ts[t] - dt_med * 0.5, BAND_MASK[0]), dt_med,
                    BAND_MASK[1] - BAND_MASK[0],
                    facecolor="black", edgecolor="none",
                ))
        _draw_face_band(ax, ts, feas_face[:, f],  y_lo=BAND_2C[0],  y_hi=BAND_2C[1])

        # SD line (Stage 2a in solid, Stage 2c in dashed for comparison).
        sd_2a_mm = raw_sd[:, f] * 1000.0
        sd_2c_mm = feas_sd[:, f] * 1000.0
        valid_2a = ~np.isnan(sd_2a_mm)
        valid_2c = ~np.isnan(sd_2c_mm)
        ax.plot(ts[valid_2a], _sd_to_norm(sd_2a_mm[valid_2a]),
                color="#222222", lw=0.8, alpha=0.8, label="2a SD")
        if valid_2c.any():
            ax.plot(ts[valid_2c], _sd_to_norm(sd_2c_mm[valid_2c]),
                    color="#c0392b", lw=0.8, alpha=0.8, linestyle="--", label="2c SD")
        # SD = 0 reference (on surface).
        ax.axhline(_sd_to_norm(0.0), color="#666666", lw=0.5, alpha=0.5, linestyle=":")

        # Axis cosmetics — labels aligned to the centre of each band.
        yticks = [
            (BAND_2A[0] + BAND_2A[1]) / 2,
            (BAND_SCH[0] + BAND_SCH[1]) / 2,
            (BAND_MASK[0] + BAND_MASK[1]) / 2,
            (BAND_2C[0] + BAND_2C[1]) / 2,
            (SD_TOP + SD_BOTTOM) / 2,
        ]
        ylabels = [
            "2a raw",
            "scheduled",
            "mask",
            "2c",
            "SD (mm)\n+30…0…−30",
        ]
        if have_manual:
            yticks.append((BAND_MANUAL[0] + BAND_MANUAL[1]) / 2)
            ylabels.append("manual")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_ylim(SD_BOTTOM - 0.1,
                    (BAND_MANUAL[1] if have_manual else BAND_2A[1]) + 0.1)
        ax.set_ylabel(fn, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=30, va="center")
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(ts[0], ts[-1])

    # Legend: face colors + line styles for the SD trace.
    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(facecolor=c, label=n)
        for c, n in zip(FACE_COLORS, FACE_NAMES)
    ] + [
        mpatches.Patch(facecolor=NO_CONTACT_COLOR, label="no contact"),
        Line2D([0], [0], color="#222222", lw=1.2, label="2a SD"),
        Line2D([0], [0], color="#c0392b", lw=1.2, ls="--", label="2c SD"),
    ]
    if have_manual:
        legend_handles.append(mpatches.Patch(facecolor="#000000", label="manual: forbid"))
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=9, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.975))

    fig.suptitle(f"Contact schedule — {session_dir.name}",
                 fontsize=12, fontweight="bold", y=0.99)

    out_path = session_dir / "contact_schedule.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--show", action="store_true",
                        help="Open interactive matplotlib window in addition to saving.")
    args = parser.parse_args()
    for d in args.session_dirs:
        print(f"\n{d.name}")
        plot_session(d, show=args.show)


if __name__ == "__main__":
    main()
