"""
Stage 1 — Contact detection on the human demonstration sequence.

For each finger, per frame, determines whether the human fingertip was in contact
with the cube based on its signed distance to the nearest cube face in cube frame.

Contact threshold: 15 mm
  = 8 mm fingertip-tip radius  +  7 mm tracking-noise headroom
  (statics calibration RMSE = 8.13 mm, see compute_calibration_metrics.py)

SPIDER stability filter (Sec. 3.2 of the paper):
  1. Drift filter: within a contact run, if the fingertip moves > 10 mm between
     consecutive frames, break the run there (tracking outlier or fast slide).
  2. Min-duration filter: discard contact runs shorter than 3 frames.

Inputs:  recordings/<session>/processed_sequence_with_proxy.npz
Output:  recordings/<session>/contact_sequence.npz
           contact_mask      (N, 5) bool   — per finger, per frame
           contact_pos_cube  (N, 5, 3)     — fingertip pos in cube frame; NaN when not in contact
           timeline          (N,)
           finger_names      saved as attribute (thumb/index/middle/ring/little)

Usage:
    python -m scripts.retargeting.spider_contact recordings/<session>
    python -m scripts.retargeting.spider_contact recordings/<session> --stats
    python -m scripts.retargeting.spider_contact recordings/<session> --force
    python -m scripts.retargeting.spider_contact recordings/cube_yaw recordings/cube_pitch
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from scripts.utils.config import CUBE_SIZE

# ── constants ─────────────────────────────────────────────────────────────────

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
N_FINGERS = 5
HALF = CUBE_SIZE / 2

# Statics-calibrated contact threshold (mm → m).
# Fingertip centre must be within this distance of the nearest cube face.
CONTACT_THRESHOLD_M = 0.015       # 15 mm

# SPIDER stability filter parameters
MIN_CONTACT_FRAMES = 3            # minimum run length kept
MAX_DRIFT_M = 0.010               # 10 mm — max fingertip displacement per frame


# ── geometry ──────────────────────────────────────────────────────────────────

def _signed_dist_to_surface(p: np.ndarray) -> float:
    """Signed distance from point p (cube frame) to the nearest cube face.

    Negative = inside the cube, zero = on surface, positive = outside.
    """
    face_signed = np.array([
        p[0] - HALF,  -p[0] - HALF,
        p[1] - HALF,  -p[1] - HALF,
        p[2] - HALF,  -p[2] - HALF,
    ])
    idx = int(np.argmin(np.abs(face_signed)))
    return float(face_signed[idx])


def _raw_contact_mask(fingertips_cube: np.ndarray) -> np.ndarray:
    """Raw contact mask before stability filtering.

    Parameters
    ----------
    fingertips_cube : (N, 5, 3) float — fingertip positions in cube frame

    Returns
    -------
    mask : (N, 5) bool
    """
    N = fingertips_cube.shape[0]
    mask = np.zeros((N, N_FINGERS), dtype=bool)
    for i in range(N):
        for f in range(N_FINGERS):
            d = _signed_dist_to_surface(fingertips_cube[i, f])
            mask[i, f] = d < CONTACT_THRESHOLD_M
    return mask


# ── SPIDER stability filter ───────────────────────────────────────────────────

def _drift_filter(mask: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Break contact runs at frames where the fingertip jumps > MAX_DRIFT_M.

    Works in-place on a copy of mask.
    """
    mask = mask.copy()
    N = mask.shape[0]
    for f in range(N_FINGERS):
        prev_pos = None
        for i in range(N):
            if mask[i, f]:
                pos = positions[i, f]
                if prev_pos is not None:
                    if np.linalg.norm(pos - prev_pos) > MAX_DRIFT_M:
                        mask[i, f] = False
                        prev_pos = None
                        continue
                prev_pos = pos
            else:
                prev_pos = None
    return mask


def _min_duration_filter(mask: np.ndarray) -> np.ndarray:
    """Remove contact runs shorter than MIN_CONTACT_FRAMES."""
    mask = mask.copy()
    for f in range(N_FINGERS):
        labeled, n = label(mask[:, f])
        for k in range(1, n + 1):
            if np.sum(labeled == k) < MIN_CONTACT_FRAMES:
                mask[labeled == k, f] = False
    return mask


def compute_contact(fingertips_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full contact detection pipeline.

    Returns
    -------
    contact_mask     : (N, 5) bool
    contact_pos_cube : (N, 5, 3) float — NaN where contact_mask is False
    """
    tips = np.asarray(fingertips_cube, dtype=float)

    raw = _raw_contact_mask(tips)
    filtered = _drift_filter(raw, tips)
    filtered = _min_duration_filter(filtered)

    contact_pos = np.full_like(tips, np.nan)
    contact_pos[filtered] = tips[filtered]

    return filtered, contact_pos


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_session(session_dir: Path):
    p = session_dir / "processed_sequence_with_proxy.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"processed_sequence_with_proxy.npz not found in {session_dir}.\n"
            "Run compute_proxy_geometry first."
        )
    return np.load(p, allow_pickle=True)


def detect_contact_session(
    session_dir: Path,
    force: bool = False,
    print_stats: bool = False,
) -> None:
    output_path = session_dir / "contact_sequence.npz"
    if output_path.exists() and not force:
        print(f"  [skip] contact_sequence.npz already exists (--force to overwrite)")
        return

    npz = _load_session(session_dir)
    timeline = np.asarray(npz["timeline"], dtype=float)
    fingertips_cube = np.asarray(npz["right_fingertips_cube"], dtype=float)  # (N, 5, 3)

    contact_mask, contact_pos_cube = compute_contact(fingertips_cube)

    np.savez_compressed(
        output_path,
        contact_mask=contact_mask,
        contact_pos_cube=contact_pos_cube,
        timeline=timeline,
        finger_names=np.array(FINGER_NAMES),
        contact_threshold_m=CONTACT_THRESHOLD_M,
        min_contact_frames=MIN_CONTACT_FRAMES,
        max_drift_m=MAX_DRIFT_M,
    )
    print(f"  Saved: {output_path}")

    if print_stats:
        _print_stats(session_dir.name, contact_mask, timeline)


def _print_stats(name: str, mask: np.ndarray, timeline: np.ndarray) -> None:
    N = mask.shape[0]
    duration_s = float(timeline[-1] - timeline[0]) if N > 1 else 0.0
    print(f"\n  {name}  ({N} frames, {duration_s:.1f} s)")
    print(f"  {'finger':<8}  {'contact%':>8}  {'runs':>5}  {'mean run':>9}")

    for f, fname in enumerate(FINGER_NAMES):
        col = mask[:, f]
        pct = 100.0 * col.mean()
        labeled, n_runs = label(col)
        run_lengths = [int(np.sum(labeled == k)) for k in range(1, n_runs + 1)]
        mean_run = float(np.mean(run_lengths)) if run_lengths else 0.0
        print(f"  {fname:<8}  {pct:>7.1f}%  {n_runs:>5}  {mean_run:>8.1f} fr")

    any_contact = mask.any(axis=1)
    all_contact = mask.all(axis=1)
    print(f"  any-finger contact: {100*any_contact.mean():.1f}%   "
          f"all-finger contact: {100*all_contact.mean():.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    force = "--force" in args
    stats = "--stats" in args
    paths = [Path(a) for a in args if not a.startswith("--")]

    if not paths:
        print(
            "Usage: python -m scripts.retargeting.spider_contact <session_dir>..."
            " [--stats] [--force]"
        )
        sys.exit(1)

    for p in paths:
        print(f"\n{p.name}")
        try:
            detect_contact_session(p, force=force, print_stats=stats)
        except Exception as e:
            print(f"  [error] {e}")


if __name__ == "__main__":
    main()
