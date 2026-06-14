"""Record and preprocess a motion capture session.

Pipeline:
    1. record_session            →  recordings/<session>/{hands,ergonomics,poses}.jsonl
    2. build_sequence_dataset    →  processed_sequence.npz
    3. compute_proxy_geometry    →  processed_sequence_with_proxy.npz

If a session directory is given, step 1 is skipped (use for reprocessing an
existing recording). Steps 2 and 3 are skipped when their outputs already
exist, unless --force is passed.

Usage:
    # Record a new session and preprocess it in one go
    python -m scripts.capture_motion

    # Reprocess an existing session
    python -m scripts.capture_motion recordings/session_20260421_150421

    # Force reprocessing even if outputs exist
    python -m scripts.capture_motion recordings/<session> --force

    # Preprocess and open the processed-sequence inspector
    python -m scripts.capture_motion recordings/<session> --inspect

    # Open the raw-stream inspector after processing
    python -m scripts.capture_motion recordings/<session> --inspect-raw
"""
import subprocess
import sys
import time
from pathlib import Path

RECORDINGS_ROOT = Path("recordings")


def run(module: str, *args: str) -> bool:
    """Run a module as a subprocess. Returns True on success."""
    cmd = ["python3", "-m", module, *args]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return False
    return True


def find_newest_session() -> Path | None:
    candidates = [
        p for p in RECORDINGS_ROOT.iterdir()
        if p.is_dir() and (p / "hands.jsonl").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def preprocess_session(session_dir: Path, force: bool) -> bool:
    status = {
        "built": (session_dir / "processed_sequence.npz").exists(),
        "proxy": (session_dir / "processed_sequence_with_proxy.npz").exists(),
    }

    ok = True
    if force or not status["built"]:
        print(f"[build]  {session_dir.name}")
        ok = run("scripts.processing.build_sequence_dataset", str(session_dir))
        if not ok:
            return False
    else:
        print(f"[skip]   {session_dir.name} — processed_sequence.npz already exists")

    if force or not status["proxy"]:
        built_path = session_dir / "processed_sequence.npz"
        print(f"[proxy]  {session_dir.name}")
        ok = run("scripts.processing.compute_proxy_geometry", str(built_path))
    else:
        print(f"[skip]   {session_dir.name} — processed_sequence_with_proxy.npz already exists")

    return ok


def main():
    args = sys.argv[1:]
    force       = "--force"       in args
    inspect     = "--inspect"     in args
    inspect_raw = "--inspect-raw" in args
    flags = {"--force", "--inspect", "--inspect-raw"}
    paths = [Path(a) for a in args if a not in flags]

    if paths:
        session_dir = paths[0]
        if not session_dir.is_dir():
            print(f"[error] {session_dir} is not a directory")
            sys.exit(1)
    else:
        print("Recording new session …")
        ok = run("scripts.processing.record_session")
        if not ok:
            sys.exit(1)
        time.sleep(0.5)
        session_dir = find_newest_session()
        if session_dir is None:
            print("[error] could not find newly recorded session in recordings/")
            sys.exit(1)
        print(f"  session: {session_dir}")

    ok = preprocess_session(session_dir, force=force)
    if not ok:
        sys.exit(1)

    if inspect_raw:
        run("scripts.visualization.inspect_capture_raw", str(session_dir))

    if inspect:
        npz = session_dir / "processed_sequence.npz"
        if npz.exists():
            run("scripts.visualization.inspect_capture_processed", str(session_dir))
        else:
            print(f"  [skip inspect] no processed_sequence.npz in {session_dir.name}")


if __name__ == "__main__":
    main()
