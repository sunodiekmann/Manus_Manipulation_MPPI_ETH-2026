"""Retarget a captured motion to the Tesollo robot hand.

Three-stage pipeline:
    1. kinematic_retargeting   →  retargeted_sequence[__<tag>].npz
    2. detect_contacts         →  contact_sequence[__<tag>].npz
    3. optimize_sequence_dynamic  →  retargeted_sequence_dynamic[__<tag>].npz

Each stage is skipped when its output already exists, unless --force is given.
Use --skip-* flags to bypass individual stages for debugging.

The --tag flag consistently suffixes all three outputs so that parameter-sweep
runs do not overwrite the canonical files. MPPI tuning knobs (--horizon,
--num-samples, --iters, --w-*, …) are intentionally absent here; pass those
directly to scripts.retargeting.optimize_sequence_dynamic when sweeping.

The --backend flag selects the Stage 3 physics rollout backend:
  cpu   — pure MuJoCo CPU rollout, always available
  mjwp  — NVIDIA Warp GPU rollout (much faster; requires CUDA 12+)

Usage:
    python -m scripts.retarget_motion recordings/<session>
    python -m scripts.retarget_motion recordings/<session> --force
    python -m scripts.retarget_motion recordings/<session> --backend mjwp
    python -m scripts.retarget_motion recordings/<session> --tag my_run
    python -m scripts.retarget_motion recordings/<session> --seed 42
    python -m scripts.retarget_motion recordings/<session> --skip-kinematic --skip-contacts
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(module: str, *args: str) -> bool:
    """Run a module as a subprocess. Returns True on success."""
    cmd = ["python3", "-m", module, *args]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="rerun stages even if their outputs already exist")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed forwarded to optimize_sequence_dynamic")
    parser.add_argument("--tag", type=str, default=None,
                        help="tag suffix for all three output files")
    parser.add_argument("--skip-kinematic", action="store_true",
                        help="skip Stage 2 kinematic retargeting")
    parser.add_argument("--skip-contacts", action="store_true",
                        help="skip Stage 2.5 contact detection")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "mjwp"],
                        help="Stage 3 rollout backend: cpu (default) or mjwp "
                             "(NVIDIA Warp GPU, requires CUDA 12+, much faster)")
    parser.add_argument("--inspect", action="store_true",
                        help="open inspect_retargeting after Stage 3 completes")
    args = parser.parse_args()

    session_dir = args.session_dir
    if not session_dir.is_dir():
        print(f"[error] {session_dir} is not a directory")
        sys.exit(1)

    force_flag = ["--force"] if args.force else []
    tag_flags  = ["--tag", args.tag] if args.tag else []

    # Stage 2 — kinematic IK
    if not args.skip_kinematic:
        print(f"\n[stage 2] kinematic retargeting — {session_dir.name}")
        ok = run("scripts.retargeting.kinematic_retargeting",
                 str(session_dir), *force_flag, *tag_flags)
        if not ok:
            sys.exit(1)
    else:
        print("\n[stage 2] skipped (--skip-kinematic)")

    # Stage 2.5 — contact detection
    if not args.skip_contacts:
        print(f"\n[stage 2.5] contact detection — {session_dir.name}")
        stage2_tag_flags = (["--stage2-tag", args.tag] if args.tag else [])
        ok = run("scripts.retargeting.detect_contacts",
                 str(session_dir), *force_flag, *stage2_tag_flags, *tag_flags)
        if not ok:
            sys.exit(1)
    else:
        print("\n[stage 2.5] skipped (--skip-contacts)")

    # Stage 3 — dynamic MPPI
    print(f"\n[stage 3] dynamic optimisation — {session_dir.name}")
    stage2_tag_flags = (["--stage2-tag", args.tag] if args.tag else [])
    seed_flags    = ["--seed", str(args.seed)] if args.seed is not None else []
    backend_flags = ["--backend", args.backend] if args.backend != "cpu" else []
    ok = run("scripts.retargeting.optimize_sequence_dynamic",
             str(session_dir), *force_flag, *seed_flags,
             *stage2_tag_flags, *tag_flags, *backend_flags)
    if not ok:
        sys.exit(1)

    # Optional inspection
    if args.inspect:
        print(f"\n[inspect] opening viewer — {session_dir.name}")
        run("scripts.visualization.inspect_retargeting",
            str(session_dir), *tag_flags)


if __name__ == "__main__":
    main()
