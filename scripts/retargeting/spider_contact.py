"""
Stage 1 — Physics-replay contact detection (SPIDER-style).

For each frame we set the Tesollo robot qpos and the cube freejoint qpos in
`assets/output_mocap_physics.xml`, run `mj_forward` (kinematic replay — no
dynamics integration), and read MuJoCo's collision detection to determine
which fingertip geoms are in contact with the cube. The fingertip body
position at the moment of contact is recorded in cube frame; per-finger
averaging across in-contact frames gives the canonical virtual-contact
target consumed by Stage 3 MPPI.

Why physics replay (vs. point-distance to cube faces):
    Stage 3 MPPI rolls out in `output_mocap_physics.xml` whose tips are boxes,
    not points. Doing detection in the same scene means contact targets and
    rollouts share the same collision geometry — no shape mismatch.

Inputs (per session):
    recordings/<session>/retargeted_sequence.npz   (Stage 2: SPIDER equality IK)
        retargeted_qpos (N, 27)  — wrist freejoint + 20 finger joints
        cube_pose_used  (N, 7)   — [x,y,z,qw,qx,qy,qz]

Output:
    recordings/<session>/contact_sequence.npz
        contact_mask         (N, 5) bool   — 3-frame OR-smoothed
        contact_pos_cube     (N, 5, 3)     — fingertip position in cube frame
                                             (NaN where contact_mask is False)
        contact_target_cube  (5, 3)        — per-finger averaged target
        contact_target_valid (5,) bool     — false if std > MAX_TARGET_STD_M
        contact_target_std   (5, 3)        — std of contact_pos_cube per finger
        timeline             (N,)
        finger_names                       — thumb/index/middle/ring/little

Usage:
    python -m scripts.retargeting.spider_contact recordings/<session>
    python -m scripts.retargeting.spider_contact recordings/cube_pitch recordings/cube_roll recordings/cube_yaw --stats
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import label

from scripts.utils.config import CUBE_SIZE

# ── constants ─────────────────────────────────────────────────────────────────

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
N_FINGERS = 5
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]
CUBE_GEOM = "tracked_cube_geom"

QPOS_HAND_LEN = 27   # 7 freejoint + 20 finger
QPOS_CUBE_LEN = 7    # 7 freejoint
QPOS_FULL_LEN = QPOS_HAND_LEN + QPOS_CUBE_LEN

# SPIDER smoothing: any contact within ±1 frame counts as contact.
SMOOTH_HALF_WINDOW = 1

# Discard a finger's averaged target if positional std exceeds this — means
# the contact location wandered too much to be a reliable virtual target.
MAX_TARGET_STD_M = 0.10


# ── physics replay ────────────────────────────────────────────────────────────

def _build_model():
    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    data = mujoco.MjData(model)

    tip_geom_ids = {
        f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for f, name in enumerate(TIP_GEOMS)
    }
    cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
    if cube_gid < 0 or any(v < 0 for v in tip_geom_ids.values()):
        raise RuntimeError("Could not resolve tip or cube geom ids in physics model.")

    geom_to_finger = {gid: f for f, gid in tip_geom_ids.items()}

    wrist_mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_mocap")
    wrist_mocap_id = int(model.body_mocapid[wrist_mocap_bid])
    if wrist_mocap_id < 0:
        raise RuntimeError("wrist_mocap is not a mocap body.")

    return model, data, tip_geom_ids, cube_gid, geom_to_finger, wrist_mocap_id


def _replay_session(retargeted_qpos: np.ndarray, cube_pose_used: np.ndarray):
    """Replay the kinematic IK trajectory and compute per-frame contact state.

    Returns
    -------
    geom_mask    : (N, 5) bool — geom collision (mj_forward) detected contact
    pos_cube     : (N, 5, 3) — fingertip body pos in cube frame, **always**
                    populated (not gated by collision). NaN only where the
                    input cube pose is NaN.
    sd_cube      : (N, 5) — signed distance of fingertip to nearest cube
                    face, in metres. Negative = inside cube, positive =
                    outside, ~0 = on surface. NaN where cube pose is NaN.
    """
    if retargeted_qpos.shape[1] != QPOS_HAND_LEN:
        raise ValueError(
            f"retargeted_qpos has {retargeted_qpos.shape[1]} cols, expected {QPOS_HAND_LEN}"
        )
    if cube_pose_used.shape[1] != QPOS_CUBE_LEN:
        raise ValueError(
            f"cube_pose_used has {cube_pose_used.shape[1]} cols, expected {QPOS_CUBE_LEN}"
        )

    model, data, tip_gid, cube_gid, geom_to_finger, wrist_mid = _build_model()

    N = retargeted_qpos.shape[0]
    geom_mask = np.zeros((N, N_FINGERS), dtype=bool)
    pos_cube = np.full((N, N_FINGERS, 3), np.nan, dtype=float)
    sd_cube = np.full((N, N_FINGERS), np.nan, dtype=float)

    qpos = np.zeros(QPOS_FULL_LEN, dtype=float)

    for i in range(N):
        if np.any(np.isnan(cube_pose_used[i])) or np.any(np.isnan(retargeted_qpos[i])):
            continue
        qpos[:QPOS_HAND_LEN] = retargeted_qpos[i]
        qpos[QPOS_HAND_LEN:] = cube_pose_used[i]
        data.qpos[:] = qpos
        data.qvel[:] = 0.0

        # Keep the wrist weld equality satisfied during forward dynamics.
        data.mocap_pos[wrist_mid] = qpos[0:3]
        data.mocap_quat[wrist_mid] = qpos[3:7]

        mujoco.mj_forward(model, data)

        # cube world pose (from forward kinematics, identical to qpos[27:34])
        cube_pos_w = data.qpos[QPOS_HAND_LEN : QPOS_HAND_LEN + 3]
        cube_quat_w = data.qpos[QPOS_HAND_LEN + 3 : QPOS_FULL_LEN]
        R_wc = np.empty(9)
        mujoco.mju_quat2Mat(R_wc, cube_quat_w)
        R_wc = R_wc.reshape(3, 3)
        in_contact = np.zeros(N_FINGERS, dtype=bool)
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == cube_gid and g2 in geom_to_finger:
                in_contact[geom_to_finger[g2]] = True
            elif g2 == cube_gid and g1 in geom_to_finger:
                in_contact[geom_to_finger[g1]] = True

        # Fingertip position in cube frame — always, regardless of contact.
        for f in range(N_FINGERS):
            tip_pos_w = np.asarray(data.geom_xpos[tip_gid[f]])
            p = R_wc.T @ (tip_pos_w - cube_pos_w)
            pos_cube[i, f] = p
            sd_cube[i, f] = float(np.abs(p).max() - _CUBE_HALF)
        geom_mask[i] = in_contact

    return geom_mask, pos_cube, sd_cube


# ── post-processing ───────────────────────────────────────────────────────────

def _smooth_mask(mask: np.ndarray, half_window: int) -> np.ndarray:
    """SPIDER's smoothing: a frame is in contact if any frame within
    ±half_window of it is in contact. (k=1 → 3-frame OR window.)"""
    out = mask.copy()
    if half_window <= 0:
        return out
    N = mask.shape[0]
    for f in range(mask.shape[1]):
        col = mask[:, f]
        for i in range(N):
            lo = max(0, i - half_window)
            hi = min(N, i + half_window + 1)
            out[i, f] = bool(col[lo:hi].any())
    return out


def _per_finger_target(pos_cube: np.ndarray, mask: np.ndarray):
    """Compute per-finger averaged contact target and validity.

    Parameters
    ----------
    pos_cube : (N, 5, 3) — NaN where mask is False
    mask     : (N, 5)   bool

    Returns
    -------
    target  : (5, 3)
    std     : (5, 3)
    valid   : (5,) bool
    """
    target = np.zeros((N_FINGERS, 3), dtype=float)
    std = np.zeros((N_FINGERS, 3), dtype=float)
    valid = np.zeros(N_FINGERS, dtype=bool)

    for f in range(N_FINGERS):
        sel = mask[:, f]
        if not sel.any():
            continue
        pts = pos_cube[sel, f]
        target[f] = pts.mean(axis=0)
        std[f] = pts.std(axis=0)
        valid[f] = bool(std[f].max() <= MAX_TARGET_STD_M)
        if not valid[f]:
            target[f] = 0.0
    return target, std, valid


MIN_PHASE_LEN = 5   # frames; shorter phases get dropped as edge-flicker noise
# After segmentation, an A→B→A pattern where the middle B-phase is shorter
# than this many frames is treated as segmentation noise (cube wobble flips
# the dominant-axis face momentarily) and absorbed back into the surrounding
# A-phase. Eliminates the back-and-forth face flipping that confuses MPPI.
MIN_ABA_MERGE_LEN = 8
# SD threshold (m) at which we consider a tip "in contact" even without a
# geom collision being detected. Catches loose pushes (Stage 2a IK lands
# 5-12 mm above the cube — collision misses, but the human intent was clearly
# contact).
SD_MASK_THRESHOLD = 0.005
# Hysteresis margin: when the previous frame's face was X, switch to face Y
# only when |coord_Y| exceeds |coord_X| by this margin. Prevents corner
# flicker where two |coords| are nearly equal.
FACE_HYSTERESIS_MARGIN_M = 0.004
_CUBE_HALF = CUBE_SIZE / 2


def _merge_short_ABA(target_per_frame: np.ndarray, min_phase_len: int) -> np.ndarray:
    """Replace short A→B→A middles with the surrounding A-phase target.

    Walks each finger's per-frame target sequence, identifies contiguous
    runs of same face id, and where a short B run is sandwiched between
    same-face A runs, rewrites the B frames to carry the A-phase's mean
    target. Idempotent under multiple passes.
    """
    N, F = target_per_frame.shape[:2]
    out = target_per_frame.copy()

    for f in range(F):
        # Build phases [start, end, face].
        phases: list[list] = []
        i = 0
        while i < N:
            p = out[i, f]
            if np.any(np.isnan(p)):
                i += 1
                continue
            face = _cube_face_id(p)
            start = i
            while i < N and not np.any(np.isnan(out[i, f])) and _cube_face_id(out[i, f]) == face:
                i += 1
            phases.append([start, i, face])

        changed = True
        while changed:
            changed = False
            j = 1
            while j < len(phases) - 1:
                pstart, pend, pface = phases[j - 1]
                cstart, cend, cface = phases[j]
                nstart, nend, nface = phases[j + 1]
                cur_len = cend - cstart
                if cur_len < min_phase_len and pface == nface:
                    a_target = target_per_frame[pstart:pend, f].mean(axis=0)
                    out[cstart:cend, f] = a_target
                    phases[j - 1] = [pstart, nend, pface]
                    del phases[j:j + 2]
                    changed = True
                else:
                    j += 1
    return out


def _cube_face_id(p: np.ndarray) -> int:
    """Return which of the 6 cube faces a contact point is on.

    Encoding: face = axis * 2 + (0 if positive, 1 if negative).
    The face is the one whose plane the point is closest to in absolute terms
    — equivalently, the axis of the largest |coordinate| component.
    """
    abs_p = np.abs(p)
    axis = int(np.argmax(abs_p))
    return axis * 2 + (0 if p[axis] >= 0.0 else 1)


def _face_ids_with_hysteresis(
    pos_cube_seq: np.ndarray,    # (N, F, 3)
    mask_seq: np.ndarray,         # (N, F) bool
    margin_m: float = FACE_HYSTERESIS_MARGIN_M,
) -> np.ndarray:
    """Per-(frame, finger) face id with hysteresis (returns -1 where not in
    contact). For each finger walks time in order: at the first in-contact
    frame after a gap or sign flip, picks argmax(|coord|). On subsequent
    frames keeps the previous face as long as its |coord| stays within
    `margin_m` of the new argmax candidate — only commits to a new face if
    the candidate clearly wins. Eliminates the corner-flicker oscillation
    that bare argmax produces when two |coords| are nearly equal.
    """
    N, F = mask_seq.shape
    face_seq = np.full((N, F), -1, dtype=int)
    for f in range(F):
        prev = -1
        for t in range(N):
            if not mask_seq[t, f] or np.any(np.isnan(pos_cube_seq[t, f])):
                prev = -1
                continue
            p = pos_cube_seq[t, f]
            abs_p = np.abs(p)
            argmax_axis = int(np.argmax(abs_p))
            argmax_face = argmax_axis * 2 + (0 if p[argmax_axis] >= 0.0 else 1)
            if prev < 0:
                face_seq[t, f] = argmax_face
                prev = argmax_face
                continue
            prev_axis = prev // 2
            prev_sign = 1.0 if (prev % 2 == 0) else -1.0
            # Sign flip on the previous axis → unambiguous switch.
            if np.sign(p[prev_axis]) != prev_sign and abs(p[prev_axis]) > 1e-6:
                face_seq[t, f] = argmax_face
                prev = argmax_face
                continue
            # Stay on prev unless a different axis wins by margin.
            if argmax_axis != prev_axis and abs_p[argmax_axis] - abs_p[prev_axis] > margin_m:
                face_seq[t, f] = argmax_face
                prev = argmax_face
            else:
                face_seq[t, f] = prev
    return face_seq


def _segment_contact_targets(
    pos_cube: np.ndarray,    # (N, 5, 3) — NaN where mask is False
    mask: np.ndarray,        # (N, 5) bool
    min_phase_len: int = MIN_PHASE_LEN,
):
    """Per-finger phases segmented by which cube face the contact is on.

    For each in-contact frame compute the dominant cube face (the one whose
    plane the contact point is closest to). A new phase starts when the face
    id changes from one in-contact frame to the next, or when a finger first
    re-enters contact after a no-contact gap. Each phase's target is the
    mean position over the phase — semantically "the finger's typical
    contact location on face X". When the cube rolls and the finger
    transitions to a new face, the phase boundary is crystal clear.

    Returns
    -------
    target_per_frame : (N, 5, 3) — NaN where the finger is not in contact
    valid_per_finger : (5,) bool — True if at least one phase exists
    n_phases         : (5,) int  — total phases per finger
    """
    N, F = mask.shape
    target_per_frame = np.full((N, F, 3), np.nan, dtype=float)
    valid_per_finger = np.zeros(F, dtype=bool)
    n_phases = np.zeros(F, dtype=int)

    # Per-(frame, finger) face id with hysteresis to suppress corner flicker.
    face_seq_all = _face_ids_with_hysteresis(pos_cube, mask)

    for f in range(F):
        col = mask[:, f]
        if not col.any():
            continue

        face_seq = face_seq_all[:, f]

        # Walk through and accumulate by face id. A phase = all in-contact
        # frames on the same cube face, possibly separated by short no-contact
        # gaps. A new phase only starts when the face id actually CHANGES (not
        # just because contact briefly dropped and resumed). Phases shorter
        # than min_phase_len are dropped as face-flicker noise.
        # Per-frame targets are assigned to the contact frames; gap frames
        # keep NaN (no pull when not in contact).
        phase_face = -2
        phase_sum = np.zeros(3, dtype=float)
        phase_n = 0
        phase_frames: list[int] = []   # explicit list of in-contact frame indices for this phase
        for t in range(N + 1):
            cur_face = face_seq[t] if t < N else -1
            face_changed = (cur_face != phase_face) and (cur_face >= 0)
            ending = (t == N)
            if face_changed or ending:
                # Close the previous phase, if any.
                if phase_face >= 0 and phase_n >= min_phase_len:
                    mean = phase_sum / phase_n
                    for pf in phase_frames:
                        target_per_frame[pf, f] = mean
                    n_phases[f] += 1
                # Start a fresh phase (only if cur_face is a real contact).
                if face_changed:
                    phase_face = cur_face
                    phase_sum = np.zeros(3, dtype=float)
                    phase_n = 0
                    phase_frames = []
            if cur_face >= 0:
                # Same face (or freshly opened): accumulate.
                phase_sum += pos_cube[t, f]
                phase_n += 1
                phase_frames.append(t)

        valid_per_finger[f] = n_phases[f] > 0

    return target_per_frame, valid_per_finger, n_phases


# ── stats ─────────────────────────────────────────────────────────────────────

def _print_stats(name: str, mask: np.ndarray, timeline: np.ndarray,
                 target: np.ndarray, std: np.ndarray, valid: np.ndarray) -> None:
    N = mask.shape[0]
    duration_s = float(timeline[-1] - timeline[0]) if N > 1 else 0.0
    print(f"\n  {name}  ({N} frames, {duration_s:.1f} s)")
    print(f"  {'finger':<8}  {'contact%':>8}  {'runs':>5}  {'mean run':>9}  "
          f"{'target (cube frame, mm)':<28}  {'std (mm)':<22}  valid")

    for f, fname in enumerate(FINGER_NAMES):
        col = mask[:, f]
        pct = 100.0 * col.mean()
        labeled, n_runs = label(col)
        run_lengths = [int(np.sum(labeled == k)) for k in range(1, n_runs + 1)]
        mean_run = float(np.mean(run_lengths)) if run_lengths else 0.0
        t_mm = target[f] * 1000.0
        s_mm = std[f] * 1000.0
        print(f"  {fname:<8}  {pct:>7.1f}%  {n_runs:>5}  {mean_run:>8.1f} fr  "
              f"({t_mm[0]:+6.1f},{t_mm[1]:+6.1f},{t_mm[2]:+6.1f})  "
              f"({s_mm[0]:5.1f},{s_mm[1]:5.1f},{s_mm[2]:5.1f})  {valid[f]}")

    any_c = mask.any(axis=1)
    all_c = mask.all(axis=1)
    print(f"  any-finger contact: {100 * any_c.mean():.1f}%   "
          f"all-finger contact: {100 * all_c.mean():.1f}%")


# ── I/O ───────────────────────────────────────────────────────────────────────

def _apply_manual_constraints(
    constraints,                # (N, 5) of {-2, -1, 0..5} from manual_contacts.load_constraints
    mask: np.ndarray,            # (N, 5) bool — pre-constraint contact mask
    pos_cube: np.ndarray,        # (N, 5, 3) — fingertip cube-frame positions
    target_per_frame: np.ndarray, # (N, 5, 3) — segmented contact target (will be patched)
):
    """Enforce manual constraints on the detection output:

    - FORBIDDEN (-2): set mask=False; clear target.
    - Face-bound (0..5): if mask is True, project the per-frame target onto
      the constrained face (the segmenter may have chosen a neighbouring
      face due to noise — manual wins).
    """
    if constraints is None:
        return mask, target_per_frame
    from scripts.retargeting.manual_contacts import FORBIDDEN, project_target_onto_face

    out_mask = mask.copy()
    out_target = target_per_frame.copy()
    N, F = mask.shape
    for t in range(N):
        for f in range(F):
            code = int(constraints[t, f])
            if code == FORBIDDEN:
                out_mask[t, f] = False
                out_target[t, f] = np.nan
                continue
            if code >= 0 and mask[t, f]:
                # If the per-frame target is NaN (segmenter dropped it for
                # being a short phase) we leave it NaN — but the mask stays.
                tgt = out_target[t, f]
                if not np.any(np.isnan(tgt)):
                    out_target[t, f] = project_target_onto_face(tgt, code, _CUBE_HALF)
    return out_mask, out_target


def detect_contact_session(
    session_dir: Path,
    force: bool = False,
    print_stats: bool = False,
    sd_mask_threshold_m: float = SD_MASK_THRESHOLD,
) -> None:
    """Run contact detection on Stage 2 (SPIDER equality-IK) output.

    The mask combines geom collision with a SD-near-surface fallback so
    loose pushes (tip 5-10 mm above the cube) still register as intended
    contact.
    """
    retarget_path = session_dir / "retargeted_sequence.npz"
    if not retarget_path.exists():
        raise FileNotFoundError(
            f"{retarget_path.name} not found in {session_dir}. "
            "Run scripts.retargeting.retarget_sequence_spider first."
        )

    output_path = session_dir / "contact_sequence.npz"
    if output_path.exists() and not force:
        print("  [skip] contact_sequence.npz already exists (--force to overwrite)")
        return

    print(f"  source: SPIDER IK  ({retarget_path.name})")
    data = np.load(retarget_path, allow_pickle=True)
    timeline = np.asarray(data["timeline"], dtype=float)
    retargeted_qpos = np.asarray(data["retargeted_qpos"], dtype=float)
    cube_pose_used = np.asarray(data["cube_pose_used"], dtype=float)

    geom_mask, pos_cube, sd_cube = _replay_session(retargeted_qpos, cube_pose_used)
    # SD-based fallback contact: tip within `sd_mask_threshold_m` of the cube
    # surface counts as contact even without a detected collision.
    sd_mask = (sd_cube <= sd_mask_threshold_m) & ~np.isnan(sd_cube)
    raw_mask = geom_mask | sd_mask
    n_geom = int(geom_mask.sum()); n_sd_only = int((sd_mask & ~geom_mask).sum())
    print(f"  contact frames: geom={n_geom}  sd-only (≤{sd_mask_threshold_m*1000:.1f} mm)={n_sd_only}")

    smoothed = _smooth_mask(raw_mask, SMOOTH_HALF_WINDOW)
    target, std, valid = _per_finger_target(pos_cube, raw_mask)

    # Per-frame face-based segmentation with hysteresis (in `_segment_contact_targets`).
    target_per_frame, valid_per_frame, n_phases = _segment_contact_targets(
        pos_cube, raw_mask,
    )
    # Collapse A→B→A segmentation noise.
    target_per_frame = _merge_short_ABA(target_per_frame, MIN_ABA_MERGE_LEN)

    # Manual constraints (optional JSON sidecar): force "no contact" or
    # "must be on face X" per (finger, time range).
    from scripts.retargeting.manual_contacts import load_constraints
    constraints = load_constraints(session_dir, timeline)
    if constraints is not None:
        from scripts.retargeting.manual_contacts import FORBIDDEN, UNCONSTRAINED
        smoothed, target_per_frame = _apply_manual_constraints(
            constraints, smoothed, pos_cube, target_per_frame,
        )
        n_forbidden = int((constraints == FORBIDDEN).sum())
        n_facebound = int((constraints >= 0).sum())
        print(f"  manual constraints: {n_forbidden} forbid-contact frames, "
              f"{n_facebound} face-bound frames")

    np.savez_compressed(
        output_path,
        contact_mask=smoothed,
        contact_pos_cube=pos_cube,
        contact_sd=sd_cube,
        contact_target_cube=target,
        contact_target_std=std,
        contact_target_valid=valid,
        contact_target_per_frame=target_per_frame,
        contact_target_per_frame_valid=valid_per_frame,
        contact_target_n_phases=n_phases,
        timeline=timeline,
        finger_names=np.array(FINGER_NAMES),
        smooth_half_window=SMOOTH_HALF_WINDOW,
        max_target_std_m=MAX_TARGET_STD_M,
        sd_mask_threshold_m=np.asarray(sd_mask_threshold_m),
        source="SPIDER IK",
    )
    print(f"  Saved: {output_path}")

    if print_stats:
        _print_stats(session_dir.name, smoothed, timeline, target, std, valid)
        print("  segmentation phases per finger (face-based): "
              + ", ".join(f"{FINGER_NAMES[f]}={int(n_phases[f])}" for f in range(N_FINGERS)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--sd-threshold", type=float, default=SD_MASK_THRESHOLD,
                        help="SD threshold (m) below which the tip is considered "
                             "in contact even without a geom collision. Catches "
                             f"loose pushes. Default {SD_MASK_THRESHOLD*1000:.1f} mm.")
    cli_args, _ = parser.parse_known_args()
    paths = cli_args.session_dirs

    if not paths:
        print(
            "Usage: python -m scripts.retargeting.spider_contact <session_dir>..."
            " [--stats] [--force] [--sd-threshold M]"
        )
        sys.exit(1)

    for p in paths:
        print(f"\n{p.name}")
        try:
            detect_contact_session(
                p,
                force=cli_args.force,
                print_stats=cli_args.stats,
                sd_mask_threshold_m=cli_args.sd_threshold,
            )
        except Exception as e:
            print(f"  [error] {e}")
            raise


if __name__ == "__main__":
    main()
