"""Manual contact-constraints — JSON sidecar that constrains the per-finger
contact face/no-contact decisions during Stage 2c IK and downstream
spider_contact.

File format (`recordings/<session>/manual_contacts.json`):

    {
      "constraints": [
        {"finger": "index",  "face": "+z",  "from_t": 0.0, "to_t": 10.0},
        {"finger": "little", "face": null,  "from_t": 0.0, "to_t": 10.0}
      ]
    }

Semantics per (finger, frame within range):
  - `face: "+z"` (etc.) — when the finger is in contact, force the contact
    target to be on that face. If the snap/contact detector would have
    classified it otherwise, override to the constrained face. Does **not**
    force contact when there isn't any.
  - `face: null` — the finger must **not** be in contact in this range.
    Stage 2c skips snap-to-surface for it; spider_contact sets mask=False.
  - Omitted time bounds → applies to the whole sequence.

Face strings → face ids (matching spider_contact._cube_face_id):
  +x=0, -x=1, +y=2, -y=3, +z=4, -z=5

Two states per (frame, finger) are produced:
  - `face[t, f]`: -1 (no constraint), -2 (forbidden contact), or 0..5
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
FACE_STR_TO_ID = {
    "+x": 0, "-x": 1,
    "+y": 2, "-y": 3,
    "+z": 4, "-z": 5,
}
FORBIDDEN = -2     # contact is explicitly forbidden
UNCONSTRAINED = -1


def load_constraints(
    session_dir: Path,
    timeline: np.ndarray,
) -> np.ndarray | None:
    """Load manual constraints for a session.

    Returns `(N, 5)` array of constraint codes, or None if no manual file
    is present. Codes:
        UNCONSTRAINED (-1) — no constraint on this (frame, finger)
        FORBIDDEN     (-2) — finger must not be in contact this frame
        0..5               — finger must be on this face if in contact

    Raises ValueError on malformed JSON.
    """
    path = session_dir / "manual_contacts.json"
    if not path.exists():
        return None

    with open(path) as f:
        spec = json.load(f)

    N = len(timeline)
    out = np.full((N, len(FINGER_NAMES)), UNCONSTRAINED, dtype=int)

    t_start = float(timeline[0])
    t_end = float(timeline[-1])

    for c in spec.get("constraints", []):
        finger = c["finger"]
        if finger not in FINGER_NAMES:
            raise ValueError(f"unknown finger: {finger!r}")
        f_idx = FINGER_NAMES.index(finger)

        face_str = c.get("face", None)
        if face_str is None:
            code = FORBIDDEN
        elif face_str in FACE_STR_TO_ID:
            code = FACE_STR_TO_ID[face_str]
        else:
            raise ValueError(
                f"unknown face: {face_str!r} (expected one of {sorted(FACE_STR_TO_ID)})"
            )

        from_t = float(c.get("from_t", t_start))
        to_t = float(c.get("to_t", t_end))
        # Find frame indices that fall in [from_t, to_t] inclusive.
        mask = (timeline >= from_t - 1e-9) & (timeline <= to_t + 1e-9)
        out[mask, f_idx] = code

    return out


def face_id_to_normal_cube(face_id: int) -> np.ndarray:
    """Outward unit normal of a face in cube frame."""
    axis = face_id // 2
    sign = 1.0 if face_id % 2 == 0 else -1.0
    n = np.zeros(3, dtype=float)
    n[axis] = sign
    return n


def project_target_onto_face(
    p_cube: np.ndarray,
    face_id: int,
    cube_half_m: float,
    inset_m: float = 0.001,
) -> np.ndarray:
    """Project a cube-frame point onto the specified cube face (SD = 0).

    Tangential coords are clipped to slightly inside the face rectangle
    (by `inset_m`) so the chosen axis is unambiguously the dominant
    |coord| — otherwise a target ending up at an edge corner like
    (-h, _, +h) is classified by argmax as either face, defeating the
    point of the constraint.
    """
    axis = face_id // 2
    sign = 1.0 if face_id % 2 == 0 else -1.0
    bound = cube_half_m - inset_m
    out = p_cube.copy()
    out[axis] = sign * cube_half_m
    for ax in range(3):
        if ax == axis:
            continue
        out[ax] = float(np.clip(out[ax], -bound, bound))
    return out


def project_target_to_nearest_face(
    p_cube: np.ndarray,
    cube_half_m: float,
    inset_m: float = 0.001,
    outward_margin_m: float = 0.0,
) -> np.ndarray:
    """Project a cube-frame point onto the nearest cube face (SD ≈ 0).

    Picks the face automatically via argmax of |coord|. Interior points
    snap to the closest face; exterior points snap to the face you'd hit
    first along the dominant axis. `outward_margin_m` shifts the result
    outward along the face normal — set to the fingertip geom half-size
    to make the commanded centre sit so the fingertip *surface* touches
    the cube (5 mm matches output_mocap_physics.xml's tip class).
    """
    axis = int(np.argmax(np.abs(p_cube)))
    sign_pos = p_cube[axis] >= 0
    face_id = 2 * axis + (0 if sign_pos else 1)
    out = project_target_onto_face(p_cube, face_id, cube_half_m, inset_m)
    if outward_margin_m != 0.0:
        out[axis] += (1.0 if sign_pos else -1.0) * outward_margin_m
    return out
