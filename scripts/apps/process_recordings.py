"""
Orchestrate the full processing pipeline for one or all recordings.

Pipeline per session:
  1. build_sequence_dataset   →  processed_sequence.npz
  2. compute_proxy_geometry   →  processed_sequence_with_proxy.npz

Each step is skipped if its output already exists.

Usage:
    # Process all sessions found under recordings/
    python -m scripts.apps.process_recordings

    # Process a single session
    python -m scripts.apps.process_recordings recordings/session_20260421_150421

    # Process and then open the processed-sequence inspector
    python -m scripts.apps.process_recordings recordings/session_20260422_105932 --inspect

    # Open the raw-stream inspector without (re)processing
    python -m scripts.apps.process_recordings recordings/session_20260422_105932 --inspect-raw

    # Force re-run even if outputs exist
    python -m scripts.apps.process_recordings recordings/session_20260421_150421 --force
"""
import subprocess
import sys
from pathlib import Path

RECORDINGS_ROOT = Path("recordings")

# ── helpers ───────────────────────────────────────────────────────────────────

def run(module: str, *args: str) -> bool:
    """Run a processing module as a subprocess. Returns True on success."""
    cmd = ["python3", "-m", module, *args]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        return False
    return True


def session_status(session_dir: Path) -> dict:
    return {
        "raw":     (session_dir / "hands.jsonl").exists(),
        "built":   (session_dir / "processed_sequence.npz").exists(),
        "proxy":   (session_dir / "processed_sequence_with_proxy.npz").exists(),
    }


def print_status(session_dir: Path, status: dict) -> None:
    def mark(ok): return "✓" if ok else "✗"
    print(
        f"  {session_dir.name}"
        f"  raw={mark(status['raw'])}"
        f"  built={mark(status['built'])}"
        f"  proxy={mark(status['proxy'])}"
    )


def process_session(session_dir: Path, force: bool = False) -> bool:
    status = session_status(session_dir)

    if not status["raw"]:
        print(f"  [SKIP] {session_dir.name} — no raw data")
        return False

    ok = True

    if force or not status["built"]:
        print(f"  [build] {session_dir.name}")
        ok = run("scripts.processing.build_sequence_dataset", str(session_dir))
        if not ok:
            return False
    else:
        print(f"  [skip]  {session_dir.name} — processed_sequence.npz already exists")

    if force or not status["proxy"]:
        built_path = session_dir / "processed_sequence.npz"
        print(f"  [proxy] {session_dir.name}")
        ok = run("scripts.processing.compute_proxy_geometry", str(built_path))
    else:
        print(f"  [skip]  {session_dir.name} — processed_sequence_with_proxy.npz already exists")

    return ok


def find_all_sessions() -> list[Path]:
    """Recursively find all session directories (contain hands.jsonl)."""
    return sorted(p.parent for p in RECORDINGS_ROOT.rglob("hands.jsonl"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    force       = "--force"       in args
    inspect     = "--inspect"     in args
    inspect_raw = "--inspect-raw" in args
    flags = {"--force", "--inspect", "--inspect-raw"}
    paths = [Path(a) for a in args if a not in flags]

    # resolve which sessions to process
    if paths:
        sessions = []
        for p in paths:
            if (p / "hands.jsonl").exists():
                sessions.append(p)
            else:
                print(f"[warn] {p} does not look like a session directory (no hands.jsonl)")
    else:
        sessions = find_all_sessions()

    if not sessions:
        print("No sessions found.")
        return

    # status overview
    print(f"Sessions found: {len(sessions)}")
    for s in sessions:
        print_status(s, session_status(s))
    print()

    # process
    if not (inspect_raw and not inspect):
        for s in sessions:
            process_session(s, force=force)

    # inspect
    if inspect_raw:
        for s in sessions:
            print(f"\nOpening raw inspector for {s.name} …")
            run("scripts.processing.inspect_raw_sequence", str(s))

    if inspect:
        for s in sessions:
            npz = s / "processed_sequence.npz"
            if npz.exists():
                print(f"\nOpening processed inspector for {s.name} …")
                run("scripts.processing.inspect_processed_sequence", str(s))
            else:
                print(f"  [skip inspect] {s.name} — no processed_sequence.npz")


if __name__ == "__main__":
    main()
