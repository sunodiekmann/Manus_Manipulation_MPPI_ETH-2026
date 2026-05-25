"""Stage 4 — open-loop verification of a Stage 3 dynamic-retargeting run.

Re-applies the saved feedforward control sequence (`ctrl_traj` +
`wrist_delta_traj`) through the physics scene WITHOUT re-planning and WITHOUT
resetting qpos between ticks, then reports how far the cube drifts from both
the reference and the originally-planned trajectory.

Why this is non-trivial
-----------------------
The MPC's saved `cube_traj` is already the feedforward outcome — `step_world`
advanced the real world continuously; re-planning only chose each action. So
re-applying the same controls on the SAME deterministic backend reproduces it
almost exactly. The test gets its signal from making replay differ from the
planning conditions:

  * cross-backend  (plan on mjwp/GPU, verify on cpu): integrator + float32→64
    differences → sensitivity to sim fidelity (sim-to-real proxy).
  * chaos / margin : contact dynamics are chaotic; if the grip has no margin
    the cube drops under replay even though it survived planning.
  * --init-pos-noise: nudge the start state and see if the open-loop controls
    still track.

Read the result as:
  * divergence small, still tracks ref  → run reproduces, low-information pass.
  * divergence large, still tracks ref  → ROBUST: many control paths work.
  * divergence large + drop             → FRAGILE: it leaned on the planner.

Usage
-----
    # verify the canonical run on cpu (cross-backend if it was planned on GPU)
    python -m scripts.evaluation.verify_open_loop recordings/cube_pitch2
    # a tagged sweep run, same backend it was planned on
    python -m scripts.evaluation.verify_open_loop recordings/cube_roll2 \
        --tag h24 --backend mjwp
    # robustness: start the cube 5 mm off
    python -m scripts.evaluation.verify_open_loop recordings/cube_pitch2 \
        --init-pos-noise 0.005 --seed 1
    # save as a dynamic-format npz so replay_dynamic/compare_dynamic can read it
    python -m scripts.evaluation.verify_open_loop recordings/cube_pitch2 \
        --save-tag ol
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CpuEnv, N_FINGER_CTRL, N_WRIST_DELTA,
)


def _build_env(backend: str, sim_dt: float, spider_physics: bool):
    if backend == "cpu":
        return CpuEnv(num_samples=1, sim_dt=sim_dt, spider_physics=spider_physics)
    if backend == "mjwp":
        from scripts.retargeting.dynamic_retargeting.mjwp_env import MjwpEnv
        return MjwpEnv(num_samples=1, sim_dt=sim_dt, spider_physics=spider_physics)
    raise ValueError(f"unknown backend: {backend}")


def _geo_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    d = abs(float(np.dot(q_a, q_b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def _stats(x: np.ndarray) -> tuple[float, float, float]:
    """mean, max, final."""
    return float(x.mean()), float(x.max()), float(x[-1])


def run(session_dir: Path, *, tag: str | None, backend: str | None,
        warmup_steps: int | None, init_pos_noise: float, seed: int,
        save_tag: str | None, drop_threshold_m: float) -> None:
    dyn_name = (f"retargeted_sequence_dynamic__{tag}.npz" if tag
                else "retargeted_sequence_dynamic.npz")
    dyn_path = session_dir / dyn_name
    if not dyn_path.exists():
        raise FileNotFoundError(f"missing {dyn_path}")
    d = np.load(dyn_path, allow_pickle=True)

    ctrl_traj = np.asarray(d["ctrl_traj"], dtype=float)             # (N, 20)
    wrist_delta_traj = np.asarray(d["wrist_delta_traj"], dtype=float)  # (N, 3)
    qpos_ref = np.asarray(d["qpos_ref_used"], dtype=float)          # (N, 27)
    cube_ref = np.asarray(d["cube_ref_used"], dtype=float)          # (N, 7)
    saved_cube = np.asarray(d["cube_traj"], dtype=float)            # (N, 7)
    timeline = np.asarray(d["timeline"], dtype=float)
    N = ctrl_traj.shape[0]

    cfg = {}
    if "run_config" in d.files:
        try:
            cfg = json.loads(str(d["run_config"]))
        except Exception:
            cfg = {}
    substeps = int(cfg.get("substeps", 25))
    sim_dt = float(cfg.get("sim_dt", 0.01))
    spider_physics = bool(cfg.get("spider_physics", True))
    planned_backend = cfg.get("backend", "unknown")
    if backend is None:
        backend = "cpu"   # default verifier = cpu (cross-checks a GPU plan)
    if warmup_steps is None:
        warmup_steps = int(cfg.get("warmup_steps", 50))

    print(f"  open-loop verify: {dyn_path.name}")
    print(f"  planned on '{planned_backend}'  →  replaying on '{backend}'"
          f"{'  (CROSS-BACKEND)' if planned_backend not in (backend, 'unknown') else ''}")
    print(f"  N={N} ticks  substeps={substeps}  sim_dt={sim_dt}"
          f"  spider_physics={spider_physics}  warmup={warmup_steps}"
          f"  init_pos_noise={init_pos_noise*1000:.1f}mm")

    env = _build_env(backend, sim_dt, spider_physics)

    # Initial state = reference start (optionally perturbed cube position).
    cube0 = cube_ref[0].copy()
    if init_pos_noise > 0.0:
        rng = np.random.default_rng(seed)
        cube0[0:3] = cube0[0:3] + rng.normal(0.0, init_pos_noise, size=3)
    env.set_world_state(qpos_ref[0], cube0)

    # Replicate the run's grip warmup so we start from the same physical grasp.
    if warmup_steps > 0:
        env.warmup_grasp(
            cube_pose7=cube0,
            finger_ctrl=qpos_ref[0, 7:27],
            ref_wrist_pos=qpos_ref[0, 0:3],
            ref_wrist_quat=qpos_ref[0, 3:7],
            n_steps=warmup_steps,
        )

    qpos_traj = np.zeros((N, 27), dtype=np.float64)
    cube_traj = np.zeros((N, 7), dtype=np.float64)
    pos_err = np.zeros(N)        # vs reference
    rot_err = np.zeros(N)        # vs reference (rad)
    div_pos = np.zeros(N)        # vs originally-planned cube_traj
    div_rot = np.zeros(N)        # vs planned (deg)

    for t in range(N):
        action = np.concatenate([ctrl_traj[t], wrist_delta_traj[t]])  # (23,)
        env.step_world(
            action=action,
            ref_wrist_pos=qpos_ref[t, 0:3],
            ref_wrist_quat=qpos_ref[t, 3:7],
            n_substeps=substeps,
        )
        cube = env.get_world_cube_pose7()
        qpos_traj[t] = env.world.qpos[:N_FINGER_CTRL + 7]
        cube_traj[t] = cube

        pos_err[t] = np.linalg.norm(cube[0:3] - cube_ref[t, 0:3])
        rot_err[t] = np.radians(_geo_deg(cube[3:7], cube_ref[t, 3:7]))
        div_pos[t] = np.linalg.norm(cube[0:3] - saved_cube[t, 0:3])
        div_rot[t] = _geo_deg(cube[3:7], saved_cube[t, 3:7])

    # ── report ──
    ref_run_pos = (np.asarray(d["pos_err_history"], dtype=float)
                   if "pos_err_history" in d.files else None)
    ref_run_rot = (np.degrees(np.asarray(d["rot_err_history"], dtype=float))
                   if "rot_err_history" in d.files else None)

    pm, pmax, pfin = _stats(pos_err * 1000)
    rm, rmax, rfin = _stats(np.degrees(rot_err))
    dpm, dpmax, dpfin = _stats(div_pos * 1000)
    drm, drmax, _ = _stats(div_rot)
    dropped = pos_err.max() > drop_threshold_m

    print("  ── tracking vs reference (open-loop) ──")
    print(f"    pos err (mm):  mean={pm:6.1f}  max={pmax:6.1f}  final={pfin:6.1f}")
    print(f"    rot err (°) :  mean={rm:6.2f}  max={rmax:6.1f}  final={rfin:6.2f}")
    if ref_run_pos is not None:
        print(f"    (planning run had: pos mean={ref_run_pos.mean()*1000:.1f}mm "
              f"max={ref_run_pos.max()*1000:.1f}mm; "
              f"rot mean={ref_run_rot.mean():.2f}° max={ref_run_rot.max():.1f}°)")
    print("  ── divergence from planned trajectory ──")
    print(f"    pos (mm):  mean={dpm:6.1f}  max={dpmax:6.1f}  final={dpfin:6.1f}")
    print(f"    rot (°) :  mean={drm:6.2f}  max={drmax:6.1f}")
    print(f"  ── verdict ──")
    # In-hand rotation is the task: a held-but-not-turning cube is a FAILURE
    # even though it never "drops". Weigh both position (drop) and orientation.
    reproduces = dpmax < 5.0 and drmax < 5.0
    rot_lost = (rfin > 30.0) or (rm > 20.0)
    if dropped:
        print(f"    DROP under open loop (max pos err {pmax:.0f}mm > "
              f"{drop_threshold_m*1000:.0f}mm) — FRAGILE: lost the cube without "
              f"the planner.")
    elif reproduces:
        print(f"    reproduces the plan (div: {dpmax:.1f}mm / {drmax:.1f}°) — "
              f"low-information pass (deterministic, same physics).")
    elif rot_lost:
        print(f"    cube HELD but ROTATION diverged (open-loop rot err "
              f"mean={rm:.0f}° final={rfin:.0f}° vs planning run "
              f"{ref_run_rot.mean():.1f}°; divergence {drmax:.0f}°). The "
              f"feedforward controls do NOT reproduce the rotation here — it "
              f"relies on per-tick replanning and/or is sim-fidelity-sensitive.")
        if planned_backend == "unknown":
            print(f"    NOTE: the run's backend is unknown (npz predates "
                  f"provenance). Re-run the verify on the OTHER backend too: if "
                  f"one reproduces and one diverges, the cause is the integrator "
                  f"gap (sim-fidelity), not feedback-dependence.")
        elif planned_backend != backend:
            print(f"    NOTE: cross-backend ({planned_backend}→{backend}). Re-run "
                  f"--backend {planned_backend} to separate the integrator gap "
                  f"from genuine feedback-dependence.")
        else:
            print(f"    Same backend → genuine feedback-dependence / chaotic-"
                  f"contact sensitivity, not just a backend gap.")
    else:
        print(f"    tracks ref despite {dpmax:.0f}mm / {drmax:.0f}° divergence "
              f"from plan — ROBUST: the control sequence holds without the planner.")

    if save_tag is not None:
        out = session_dir / f"retargeted_sequence_dynamic__{save_tag}.npz"
        save_cfg = dict(cfg)
        save_cfg.update({"tag": save_tag, "mode": "open_loop_verify",
                         "verify_backend": backend, "planned_backend": planned_backend,
                         "init_pos_noise_m": init_pos_noise})
        np.savez_compressed(
            out,
            timeline=timeline,
            retargeted_qpos=qpos_traj,
            cube_traj=cube_traj,
            ctrl_traj=ctrl_traj,
            wrist_delta_traj=wrist_delta_traj,
            pos_err_history=pos_err,
            rot_err_history=rot_err,
            iter_improvement_history=np.zeros(N),
            cube_ref_used=cube_ref,
            qpos_ref_used=qpos_ref,
            run_config=np.array(json.dumps(save_cfg)),
        )
        print(f"  saved (dynamic-format) → {out.name}  "
              f"(replay_dynamic --tag {save_tag} / compare_dynamic)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session_dir", type=Path)
    p.add_argument("--tag", type=str, default=None,
                   help="which dynamic run to verify (default: canonical)")
    p.add_argument("--backend", choices=["cpu", "mjwp"], default=None,
                   help="replay backend (default cpu — cross-checks a GPU plan)")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="grip warmup substeps (default: same as the run)")
    p.add_argument("--init-pos-noise", type=float, default=0.0,
                   help="std (m) of gaussian noise on the initial cube position")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-tag", type=str, default=None,
                   help="save the open-loop rollout as "
                        "retargeted_sequence_dynamic__<save-tag>.npz")
    p.add_argument("--drop-threshold", type=float, default=0.10)
    args = p.parse_args()
    run(args.session_dir, tag=args.tag, backend=args.backend,
        warmup_steps=args.warmup_steps, init_pos_noise=args.init_pos_noise,
        seed=args.seed, save_tag=args.save_tag,
        drop_threshold_m=args.drop_threshold)


if __name__ == "__main__":
    main()
