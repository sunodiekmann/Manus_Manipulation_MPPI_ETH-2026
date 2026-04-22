"""
Compute calibration error for static-pose recordings.

For each session in STATICS_DIR:
  1. Builds processed_sequence.npz (if missing)
  2. Builds processed_sequence_with_proxy.npz (if missing)
  3. Computes per-fingertip distance to the nearest cube face

Usage:
    python -m scripts.evaluation.compute_calibration_metrics
    python -m scripts.evaluation.compute_calibration_metrics recordings/statics
"""
import sys
import subprocess
from pathlib import Path

import numpy as np

from scripts.utils.config import CUBE_SIZE

STATICS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("recordings/statics")
FINGERTIP_RADIUS = 0.008  # meters — should match optimize_hand_object_alignment.py

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
HALF = CUBE_SIZE / 2


# ── geometry ─────────────────────────────────────────────────────────────────

def signed_dist_to_surface(p: np.ndarray) -> tuple[float, str]:
    """
    Signed distance from point p (in cube frame) to the nearest cube face.
      < 0 → point is inside the cube (error: too deep)
      = 0 → point is exactly on the surface
      > 0 → point is outside the cube (error: floating away)

    Returns (signed_distance_m, face_label).
    """
    face_labels = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    face_signed = np.array([
        p[0] - HALF,
        -p[0] - HALF,
        p[1] - HALF,
        -p[1] - HALF,
        p[2] - HALF,
        -p[2] - HALF,
    ])
    idx = int(np.argmin(np.abs(face_signed)))
    return float(face_signed[idx]), face_labels[idx]


# ── pipeline helpers ──────────────────────────────────────────────────────────

def ensure_processed(session_dir: Path) -> Path:
    out = session_dir / "processed_sequence.npz"
    if not out.exists():
        print(f"  [build] {session_dir.name} …", flush=True)
        subprocess.run(
            ["python3", "-m", "scripts.processing.build_sequence_dataset", str(session_dir)],
            check=True,
        )
    return out


def ensure_proxy(session_dir: Path) -> Path:
    out = session_dir / "processed_sequence_with_proxy.npz"
    if not out.exists():
        processed = session_dir / "processed_sequence.npz"
        print(f"  [proxy] {session_dir.name} …", flush=True)
        subprocess.run(
            ["python3", "-m", "scripts.processing.compute_proxy_geometry", str(processed)],
            check=True,
        )
    return out


# ── per-session metrics ───────────────────────────────────────────────────────

def session_metrics(proxy_path: Path) -> dict:
    data = np.load(proxy_path, allow_pickle=True)

    left_tips_cube  = list(data["left_fingertips_cube"])
    right_tips_cube = list(data["right_fingertips_cube"])

    def collect(tips_seq):
        errors, faces = [], []
        for tips in tips_seq:
            if tips is None:
                continue
            tips = np.asarray(tips, dtype=float)
            row_e, row_f = [], []
            for fi in range(5):
                d, face = signed_dist_to_surface(tips[fi])
                row_e.append(d)
                row_f.append(face)
            errors.append(row_e)
            faces.append(row_f)
        return np.array(errors, dtype=float), faces

    left_errors,  left_faces  = collect(left_tips_cube)
    right_errors, right_faces = collect(right_tips_cube)

    return {
        "left_errors":  left_errors,
        "right_errors": right_errors,
        "left_faces":   left_faces,
        "right_faces":  right_faces,
    }


# ── printing ─────────────────────────────────────────────────────────────────

def print_hand_summary(name: str, errors: np.ndarray, faces: list) -> None:
    if errors.size == 0:
        print(f"  {name}: no valid frames")
        return

    print(f"  {name}  ({len(errors)} frames,  cube half={HALF*100:.1f} cm,  tip radius={FINGERTIP_RADIUS*100:.1f} cm)")
    print(f"    {'finger':<8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'nearest face'}")
    print(f"    {'':8}  {'(mm)':>8}  {'(mm)':>8}  {'(mm)':>8}  {'(mm)':>8}")

    for fi, fname in enumerate(FINGER_NAMES):
        col = errors[:, fi] * 1000  # m → mm
        face_col = [row[fi] for row in faces]
        face_counts: dict[str, int] = {}
        for f in face_col:
            face_counts[f] = face_counts.get(f, 0) + 1
        dominant_face = max(face_counts, key=face_counts.get)
        print(f"    {fname:<8}  {col.mean():>+8.2f}  {col.std():>8.2f}  {col.min():>+8.2f}  {col.max():>+8.2f}  {dominant_face}")

    all_col = errors.flatten() * 1000
    abs_col = np.abs(all_col)
    print(f"    {'ALL':<8}  {all_col.mean():>+8.2f}  {all_col.std():>8.2f}  {all_col.min():>+8.2f}  {all_col.max():>+8.2f}")
    print(f"    mean |error| = {abs_col.mean():.2f} mm   RMSE = {np.sqrt((all_col**2).mean()):.2f} mm")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sessions = sorted(p for p in STATICS_DIR.iterdir() if p.is_dir())
    if not sessions:
        print(f"No sessions found in {STATICS_DIR}")
        return

    print(f"Calibration error report — {STATICS_DIR}")
    print(f"Metric: signed distance fingertip centre → nearest cube face")
    print(f"  +value = outside cube  |  −value = inside cube  |  0 = on surface")
    print()

    all_left, all_right = [], []

    for session_dir in sessions:
        print(f"{'=' * 60}")
        print(f"Session: {session_dir.name}")

        try:
            ensure_processed(session_dir)
            proxy_path = ensure_proxy(session_dir)
            m = session_metrics(proxy_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print_hand_summary("LEFT  hand", m["left_errors"],  m["left_faces"])
        print()
        print_hand_summary("RIGHT hand", m["right_errors"], m["right_faces"])
        print()

        if m["left_errors"].size > 0:
            all_left.append(m["left_errors"])
        if m["right_errors"].size > 0:
            all_right.append(m["right_errors"])

    if len(sessions) > 1:
        print(f"{'=' * 60}")
        print("AGGREGATE (all sessions combined)")
        print()
        if all_left:
            combined = np.vstack(all_left)
            print_hand_summary("LEFT  hand", combined, [["?"] * 5] * len(combined))
        if all_right:
            combined = np.vstack(all_right)
            print_hand_summary("RIGHT hand", combined, [["?"] * 5] * len(combined))
        print()


if __name__ == "__main__":
    main()
