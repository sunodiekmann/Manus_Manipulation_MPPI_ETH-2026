"""Visualize per-finger contact schedule over time.

For each of the 5 fingers, plots a stack of bands sharing a common x-axis:

  - raw face      : FK-based nearest cube-face id for the Stage 2 (SPIDER IK)
                    tip. "If this finger touched something now, which face
                    would it touch?" Doesn't say whether contact is actually
                    happening; just the dominant cube-frame axis.
  - scheduled face: detect_contacts's per-frame segmented contact target,
                    coloured. Grey where the finger is not flagged as in
                    contact (NaN in contact_target_per_frame).
  - mask          : black where detect_contacts's contact_mask is True (tip
                    geom collided with cube geom in mj_forward).
  - SD (mm)       : signed distance of the FK tip to the nearest cube face,
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

from scripts.retargeting.detect_contacts import _cube_face_id
from scripts.utils.config import CUBE_SIZE
from scripts.utils.transforms import pose_to_matrix

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]

# Stable, distinct colours per cube face. Pairs of related hues for ±axis.
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
    """Draw a horizontal coloured band: a rect per contiguous same-face run.
    -1 face id → no-contact grey. Bands span [y_lo, y_hi] vertically.
    """
    N = len(face_seq)
    if N == 0:
        return
    dt = float(np.median(np.diff(ts))) if N > 1 else 1.0
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
    rt_path = session_dir / "retargeted_sequence.npz"
    if not cs_path.exists():
        raise FileNotFoundError(cs_path)
    if not rt_path.exists():
        raise FileNotFoundError(rt_path)

    cs = np.load(cs_path)
    rt = np.load(rt_path)
    timeline = np.asarray(cs["timeline"], dtype=float)
    target_pf = np.asarray(cs["contact_target_per_frame"], dtype=float)
    contact_mask = np.asarray(cs["contact_mask"], dtype=bool)

    cleaned = _cleaned_face_per_frame(target_pf)

    print(f"  computing FK on Stage 2 qpos ({len(timeline)} frames) …")
    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    raw_face, raw_sd = _compute_raw_face_and_sd(
        model,
        np.asarray(rt["retargeted_qpos"], dtype=float),
        np.asarray(rt["cube_pose_used"], dtype=float),
    )

    ts = timeline - timeline[0]  # seconds from start
    N = len(ts)

    fig, axes = plt.subplots(
        5, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    # Band y-coordinates (stacked top-to-bottom in each finger subplot).
    BAND_RAW = (3.2, 3.8)        # FK raw face id
    BAND_SCH = (2.4, 3.0)        # scheduled face id
    BAND_MASK = (2.05, 2.20)     # mask strip
    SD_TOP = 1.6                 # SD line band top
    SD_BOTTOM = 0.4              # SD line band bottom
    SD_SCALE_MIN, SD_SCALE_MAX = -30.0, 30.0  # mm shown
    dt_med = float(np.median(np.diff(ts))) if N > 1 else 1.0

    def _sd_to_norm(mm):
        frac = (np.clip(mm, SD_SCALE_MIN, SD_SCALE_MAX) - SD_SCALE_MIN) / (SD_SCALE_MAX - SD_SCALE_MIN)
        return SD_TOP - frac * (SD_TOP - SD_BOTTOM)

    for f, (ax, fn) in enumerate(zip(axes, FINGER_NAMES)):
        _draw_face_band(ax, ts, raw_face[:, f], y_lo=BAND_RAW[0], y_hi=BAND_RAW[1])
        _draw_face_band(ax, ts, cleaned[:, f],  y_lo=BAND_SCH[0], y_hi=BAND_SCH[1])
        for t in range(N):
            if contact_mask[t, f]:
                ax.add_patch(mpatches.Rectangle(
                    (ts[t] - dt_med * 0.5, BAND_MASK[0]), dt_med,
                    BAND_MASK[1] - BAND_MASK[0],
                    facecolor="black", edgecolor="none",
                ))

        sd_mm = raw_sd[:, f] * 1000.0
        valid = ~np.isnan(sd_mm)
        ax.plot(ts[valid], _sd_to_norm(sd_mm[valid]),
                color="#222222", lw=0.8, alpha=0.8)
        ax.axhline(_sd_to_norm(0.0), color="#666666", lw=0.5, alpha=0.5, linestyle=":")

        yticks = [
            (BAND_RAW[0] + BAND_RAW[1]) / 2,
            (BAND_SCH[0] + BAND_SCH[1]) / 2,
            (BAND_MASK[0] + BAND_MASK[1]) / 2,
            (SD_TOP + SD_BOTTOM) / 2,
        ]
        ylabels = ["raw", "scheduled", "mask", "SD (mm)\n+30…0…−30"]
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_ylim(SD_BOTTOM - 0.1, BAND_RAW[1] + 0.1)
        ax.set_ylabel(fn, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=30, va="center")
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(ts[0], ts[-1])

    legend_handles = [
        mpatches.Patch(facecolor=c, label=n) for c, n in zip(FACE_COLORS, FACE_NAMES)
    ] + [
        mpatches.Patch(facecolor=NO_CONTACT_COLOR, label="no contact"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.975))

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
