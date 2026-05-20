"""Stage 3 — Dynamic retargeting via SPIDER-style sampling MPC.

Reads Stage 2 (`retargeted_sequence.npz`) + Stage 2.5 contact targets
(`contact_sequence.npz`) and produces a dynamically feasible trajectory
by rolling MPPI in the base physics scene (cube as a real freejoint,
no virtual stick — fingers must grip and rotate the cube under physics).

Action layout (per step): [20 finger ctrls, 3 wrist axis-angle delta].
Wrist policy: position locked to ref; orientation = ref ⊗ exp(action delta).

Inputs (per session):
    recordings/<session>/retargeted_sequence.npz   (Stage 2: SPIDER equality IK)
    recordings/<session>/contact_sequence.npz      (Stage 2.5: spider_contact)

Output:
    recordings/<session>/retargeted_sequence_dynamic.npz
        timeline (N,)
        retargeted_qpos (N, 27)        — sim hand qpos after each MPC tick
        cube_traj (N, 7)               — sim cube pose7 (pos + wxyz)
        ctrl_traj (N, 20)              — applied finger controls
        wrist_delta_traj (N, 3)        — applied wrist axis-angle delta
        rew_max, rew_mean              — MPPI diagnostics per tick
        pos_err_history, rot_err_history, iter_improvement_history
        term_names, term_means_history — per-term reward diagnostics

Usage:
    python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session>
    python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session> --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CpuEnv,
    N_ACTION,
    N_FINGER_CTRL,
    N_WRIST_DELTA,
)
from scripts.retargeting.dynamic_retargeting.mppi import MPPIConfig, optimize_step
from scripts.retargeting.dynamic_retargeting.reward import RewardWeights, TERM_NAMES


def _build_env(backend: str, num_samples: int):
    if backend == "cpu":
        return CpuEnv(num_samples=num_samples)
    if backend == "mjwp":
        from scripts.retargeting.dynamic_retargeting.mjwp_env import MjwpEnv
        return MjwpEnv(num_samples=num_samples)
    raise ValueError(f"unknown backend: {backend}")


def _pad_reference(arr: np.ndarray, total_len: int) -> np.ndarray:
    """Pad along axis 0 by repeating the last frame to `total_len` rows."""
    if arr.shape[0] >= total_len:
        return arr[:total_len]
    pad = np.repeat(arr[-1:], total_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _resample_reference(
    K: int,
    qpos_ref: np.ndarray,                  # (N, 27) wrist freejoint + 20 fingers
    cube_ref: np.ndarray,                  # (N, 7)  [x,y,z, qw,qx,qy,qz]
    contact_target: np.ndarray | None,     # (N, 5, 3) cube-frame, NaN allowed
    timeline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Insert K-1 interpolated frames between each pair of input frames so
    the per-tick reference delta is K× smaller (i.e. slower reference
    motion at the same sim cadence). The total task duration grows K×.

    Quaternions use SLERP; everything else linear. Contact-target entries
    that have NaN at either endpoint of an interval stay NaN in the
    interpolated region.
    """
    K = int(K)
    if K <= 1:
        return qpos_ref, cube_ref, contact_target, timeline
    from scipy.spatial.transform import Rotation as R, Slerp
    N = qpos_ref.shape[0]
    M = (N - 1) * K + 1
    t_in = np.arange(N, dtype=float)
    t_out = np.linspace(0.0, N - 1, M)

    new_qpos = np.zeros((M, 27), dtype=np.float64)
    for j in (0, 1, 2, *range(7, 27)):
        new_qpos[:, j] = np.interp(t_out, t_in, qpos_ref[:, j])
    wrist_xyzw = qpos_ref[:, [4, 5, 6, 3]]
    wq_out_xyzw = Slerp(t_in, R.from_quat(wrist_xyzw))(t_out).as_quat()
    new_qpos[:, 3] = wq_out_xyzw[:, 3]
    new_qpos[:, 4:7] = wq_out_xyzw[:, :3]

    new_cube = np.zeros((M, 7), dtype=np.float64)
    for j in range(3):
        new_cube[:, j] = np.interp(t_out, t_in, cube_ref[:, j])
    cube_xyzw = cube_ref[:, [4, 5, 6, 3]]
    cq_out_xyzw = Slerp(t_in, R.from_quat(cube_xyzw))(t_out).as_quat()
    new_cube[:, 3] = cq_out_xyzw[:, 3]
    new_cube[:, 4:7] = cq_out_xyzw[:, :3]

    new_contact = None
    if contact_target is not None:
        F = contact_target.shape[1]
        new_contact = np.full((M, F, 3), np.nan, dtype=np.float64)
        for f in range(F):
            for c in range(3):
                vals = contact_target[:, f, c]
                valid = ~np.isnan(vals)
                for i in range(N - 1):
                    if not (valid[i] and valid[i + 1]):
                        continue
                    sel = (t_out >= t_in[i]) & (t_out <= t_in[i + 1])
                    alpha = (t_out[sel] - t_in[i]) / (t_in[i + 1] - t_in[i])
                    new_contact[sel, f, c] = (1 - alpha) * vals[i] + alpha * vals[i + 1]
                if valid[0]:
                    new_contact[0, f, c] = vals[0]
                if valid[-1]:
                    new_contact[-1, f, c] = vals[-1]

    dt_in = float(np.diff(timeline).mean()) if N > 1 else 1.0
    new_timeline = timeline[0] + dt_in * np.arange(M)
    return new_qpos, new_cube, new_contact, new_timeline


def run_session(
    session_dir: Path,
    *, force: bool, cfg: MPPIConfig, seed: int,
    warmup_steps: int = 50, warmup_closure: float = 0.0,
    loop_closure: float = 0.0,
    backend: str = "cpu",
    time_scale: int = 1,
    max_ticks: int = 0,
    stale_rotation_window_s: float = 0.0,
    stale_rotation_deg: float = 5.0,
) -> None:
    retarget_path = session_dir / "retargeted_sequence.npz"
    contact_path = session_dir / "contact_sequence.npz"
    out_path = session_dir / "retargeted_sequence_dynamic.npz"

    if not retarget_path.exists():
        raise FileNotFoundError(
            f"missing {retarget_path.name}. "
            "Run scripts.retargeting.retarget_sequence_spider first.")
    if not contact_path.exists():
        raise FileNotFoundError(f"missing {contact_path}")
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (--force to overwrite)")
        return

    print(f"  Stage 2 plan: {retarget_path.name}")
    rt = np.load(retarget_path)
    timeline = np.asarray(rt["timeline"], dtype=float)
    qpos_ref = np.asarray(rt["retargeted_qpos"], dtype=float)        # (N, 27)
    cube_ref = np.asarray(rt["cube_pose_used"], dtype=float)         # (N, 7)
    if "cube_shift" in rt.files:
        cs = np.asarray(rt["cube_shift"], dtype=float)
        print(f"  Stage 2 baked-in cube_shift (mm): "
              f"[{cs[0]*1000:+.1f}, {cs[1]*1000:+.1f}, {cs[2]*1000:+.1f}]")
    cm = np.load(contact_path)
    contact_target_per_frame = (
        np.asarray(cm["contact_target_per_frame"], dtype=float)
        if "contact_target_per_frame" in cm.files else None
    )

    if time_scale > 1:
        N_orig = qpos_ref.shape[0]
        qpos_ref, cube_ref, contact_target_per_frame, timeline = _resample_reference(
            time_scale, qpos_ref, cube_ref, contact_target_per_frame, timeline)
        print(f"  time-scale: {time_scale}× (reference upsampled "
              f"{N_orig} → {qpos_ref.shape[0]} frames; per-tick Δref / {time_scale}, "
              f"total task time × {time_scale})")

    N = qpos_ref.shape[0]
    H = cfg.horizon_steps
    qpos_ref_padded = _pad_reference(qpos_ref, N + H)
    cube_ref_padded = _pad_reference(cube_ref, N + H)
    if contact_target_per_frame is not None:
        last = contact_target_per_frame[-1:].copy()
        last[:] = np.nan
        pad = np.repeat(last, (N + H) - contact_target_per_frame.shape[0], axis=0)
        contact_target_per_frame_padded = np.concatenate(
            [contact_target_per_frame, pad], axis=0)
    else:
        contact_target_per_frame_padded = None

    print(f"  sequence: {N} ref ticks  ({timeline[-1]-timeline[0]:.1f}s)")
    print(f"  MPPI: N={cfg.num_samples}, H={cfg.horizon_steps}, iters={cfg.max_iters},"
          f" substeps={cfg.n_substeps}")
    print(f"  noise std: finger={cfg.finger_noise_std}  wrist={cfg.wrist_noise_std}")

    print(f"  backend: {backend}")
    env = _build_env(backend, cfg.num_samples)
    env.set_world_state(qpos_ref[0], cube_ref[0])
    if loop_closure > 0.0:
        env.set_closure_bias(loop_closure)

    # Kinematic-pin warmup: pin cube at ref[0], step physics so fingers
    # settle into a grip. After this the cube is released and runs free.
    if warmup_steps > 0:
        env.warmup_grasp(
            cube_pose7=cube_ref[0],
            finger_ctrl=qpos_ref[0, 7:27],
            ref_wrist_pos=qpos_ref[0, 0:3],
            ref_wrist_quat=qpos_ref[0, 3:7],
            n_steps=warmup_steps,
            closure_bias=warmup_closure,
        )
        print(f"  warmup grasp: {warmup_steps} substeps with cube pinned  "
              f"closure_bias={warmup_closure:.3f} rad")

    rng = np.random.default_rng(seed)

    qpos_traj = np.zeros((N, 27), dtype=np.float64)
    cube_traj = np.zeros((N, 7), dtype=np.float64)
    ctrl_traj = np.zeros((N, N_FINGER_CTRL), dtype=np.float64)
    wrist_delta_traj = np.zeros((N, N_WRIST_DELTA), dtype=np.float64)
    rew_max_log, rew_mean_log = [], []
    pos_err_history = np.zeros(N, dtype=np.float64)
    rot_err_history = np.zeros(N, dtype=np.float64)
    iter_improvement_history = np.zeros(N, dtype=np.float64)
    term_means_history = np.zeros((N, len(TERM_NAMES)), dtype=np.float64)

    # Initial plan: finger ctrls = IK ref, wrist delta = 0.
    plan = np.zeros((H, N_ACTION), dtype=np.float64)
    plan[:, :N_FINGER_CTRL] = qpos_ref_padded[:H, 7:27]
    plan[:, N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0

    N_run = N if max_ticks <= 0 else min(N, max_ticks)
    if N_run < N:
        print(f"  capped to {N_run}/{N} MPC ticks (--max-ticks)")

    t_start = time.perf_counter()
    for t in range(N_run):
        ref_qpos_h = qpos_ref_padded[t : t + H]
        ref_cube_pos_h = cube_ref_padded[t : t + H, 0:3]
        ref_cube_quat_h = cube_ref_padded[t : t + H, 3:7]
        ref_wrist_pos_h = qpos_ref_padded[t : t + H, 0:3]
        ref_wrist_quat_h = qpos_ref_padded[t : t + H, 3:7]
        contact_target_h = (
            contact_target_per_frame_padded[t : t + H]
            if contact_target_per_frame_padded is not None else None
        )

        plan, info = optimize_step(
            env=env,
            plan=plan,
            ref_qpos=ref_qpos_h,
            ref_cube_pos=ref_cube_pos_h,
            ref_cube_quat=ref_cube_quat_h,
            ref_wrist_pos=ref_wrist_pos_h,
            ref_wrist_quat=ref_wrist_quat_h,
            contact_target_per_step=contact_target_h,
            cfg=cfg,
            rng=rng,
        )

        action = plan[0]
        env.step_world(
            action=action,
            ref_wrist_pos=ref_wrist_pos_h[0],
            ref_wrist_quat=ref_wrist_quat_h[0],
            n_substeps=cfg.n_substeps,
        )

        qpos_traj[t] = env.world.qpos[:N_FINGER_CTRL + 7]
        cube_traj[t] = env.get_world_cube_pose7()
        ctrl_traj[t] = action[:N_FINGER_CTRL]
        wrist_delta_traj[t] = action[N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA]
        rew_max_log.append(info["rew_max"])
        rew_mean_log.append(info["rew_mean"])

        sim_cube_pose = cube_traj[t]
        ref_cube_pose_now = cube_ref_padded[t]
        pos_err_history[t] = float(np.linalg.norm(
            sim_cube_pose[0:3] - ref_cube_pose_now[0:3]))
        dot = float(abs(np.dot(sim_cube_pose[3:7], ref_cube_pose_now[3:7])))
        rot_err_history[t] = float(2.0 * np.arccos(min(1.0, max(-1.0, dot))))
        iter_improvement_history[t] = (
            info["rew_max"][-1] - info["rew_max"][0])
        for ti, k in enumerate(TERM_NAMES):
            term_means_history[t, ti] = info["terms"][k][-1]

        # Slide horizon: drop applied step, append next IK ref tail.
        plan = np.concatenate(
            [plan[1:], np.zeros((1, N_ACTION), dtype=np.float64)], axis=0)
        plan[-1, :N_FINGER_CTRL] = qpos_ref_padded[t + H, 7:27]
        plan[-1, N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0

        if t % 10 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (t + 1) / elapsed
            eta = (N_run - t - 1) / rate if rate > 0 else 0.0
            print(f"  t={t:4d}/{N}  rew={info['rew_max'][-1]:+7.2f}"
                  f" (improv {iter_improvement_history[t]:+5.2f})  "
                  f"pos={pos_err_history[t]*1000:5.1f}mm  "
                  f"rot={np.degrees(rot_err_history[t]):5.1f}°  "
                  f"frac_term={info['frac_terminated'][-1]:.2f}  "
                  f"rate={rate:.2f} t/s  ETA={eta:.0f}s")
            terms_str = "  ".join(
                f"{k}={info['terms'][k][-1]:+6.2f}" for k in TERM_NAMES)
            print(f"           terms: {terms_str}")

        # Stale-rotation early stop.
        if stale_rotation_window_s > 0:
            dt = float(np.diff(timeline).mean()) if len(timeline) > 1 else 0.05
            win = max(2, int(stale_rotation_window_s / dt))
            if t >= win:
                sim_q0 = cube_traj[t - win, 3:7]
                sim_q1 = cube_traj[t, 3:7]
                ref_q0 = cube_ref[t - win, 3:7]
                ref_q1 = cube_ref[t, 3:7]
                sim_rot = np.degrees(2 * np.arccos(np.clip(
                    abs(np.dot(sim_q0, sim_q1)), -1, 1)))
                ref_rot = np.degrees(2 * np.arccos(np.clip(
                    abs(np.dot(ref_q0, ref_q1)), -1, 1)))
                if sim_rot < stale_rotation_deg and ref_rot > 3 * stale_rotation_deg:
                    print(f"  [terminated] stale rotation at t={t}: "
                          f"sim rotated {sim_rot:.1f}° in last {stale_rotation_window_s}s "
                          f"vs ref {ref_rot:.1f}°. MPC is stuck.")
                    cube_traj = cube_traj[: t + 1]
                    qpos_traj = qpos_traj[: t + 1]
                    ctrl_traj = ctrl_traj[: t + 1]
                    wrist_delta_traj = wrist_delta_traj[: t + 1]
                    timeline = timeline[: t + 1]
                    break

    elapsed = time.perf_counter() - t_start
    print(f"  Done in {elapsed:.1f}s ({len(timeline)/elapsed:.2f} ticks/s).")

    K = min(len(timeline), N_run, len(rew_max_log))
    timeline = timeline[:K]
    qpos_traj = qpos_traj[:K]
    cube_traj = cube_traj[:K]
    ctrl_traj = ctrl_traj[:K]
    wrist_delta_traj = wrist_delta_traj[:K]
    rew_max_log = rew_max_log[:K]
    rew_mean_log = rew_mean_log[:K]
    # Save the (post-resample) reference cube + hand that MPC was actually
    # optimizing against. Replay reads these for the ghost overlay so the
    # frame count matches the dynamic trajectory even with --time-scale.
    cube_ref_used = cube_ref_padded[:K].copy()
    qpos_ref_used = qpos_ref_padded[:K].copy()
    np.savez_compressed(
        out_path,
        timeline=timeline,
        retargeted_qpos=qpos_traj,
        cube_traj=cube_traj,
        ctrl_traj=ctrl_traj,
        wrist_delta_traj=wrist_delta_traj,
        rew_max=np.asarray(rew_max_log),
        rew_mean=np.asarray(rew_mean_log),
        pos_err_history=pos_err_history[:K],
        rot_err_history=rot_err_history[:K],
        iter_improvement_history=iter_improvement_history[:K],
        term_names=np.asarray(list(TERM_NAMES)),
        term_means_history=term_means_history[:K],
        cube_ref_used=cube_ref_used,
        qpos_ref_used=qpos_ref_used,
    )
    print(f"  ── error summary ──")
    print(f"  cube pos err (mm):  mean={pos_err_history[:K].mean()*1000:5.1f}  "
          f"max={pos_err_history[:K].max()*1000:5.1f}")
    print(f"  cube rot err (°):   mean={np.degrees(rot_err_history[:K]).mean():5.1f}  "
          f"max={np.degrees(rot_err_history[:K]).max():5.1f}")
    print(f"  iter improvement:   mean={iter_improvement_history[:K].mean():+5.2f}  "
          f"max={iter_improvement_history[:K].max():+5.2f}")
    print(f"  mean per-term reward across ticks (more-negative = bigger penalty):")
    for ti, k in enumerate(TERM_NAMES):
        print(f"    {k:12s} {term_means_history[:K, ti].mean():+7.3f}")
    print(f"  Saved: {out_path}")


def _build_weights(args) -> RewardWeights:
    presets = {
        "default": RewardWeights(),
        "object_heavy": RewardWeights(
            cube_pos=3.0, cube_quat=1.0, finger_qpos=1.0,
            contact=1.0, wrist_delta=0.5,
        ),
    }
    w = presets.get(args.preset)
    if w is None:
        raise SystemExit(f"unknown --preset: {args.preset}; choices: {list(presets)}")
    for field in ("cube_pos", "cube_quat", "finger_qpos", "contact", "wrist_delta", "qvel"):
        v = getattr(args, f"w_{field}", None)
        if v is not None:
            setattr(w, field, v)
    if args.terminal_scale is not None:
        w.terminal_scale = float(args.terminal_scale)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=25)
    parser.add_argument("--preset", default="default",
                        choices=["default", "object_heavy"])
    parser.add_argument("--w-cube-pos", type=float, default=None)
    parser.add_argument("--w-cube-quat", type=float, default=None)
    parser.add_argument("--w-finger-qpos", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--w-wrist-delta", type=float, default=None)
    parser.add_argument("--w-qvel", type=float, default=None,
                        help="Velocity penalty weight (per rad/s, m/s). "
                             "Encourages stable equilibrium states. Default 0.005.")
    parser.add_argument("--terminal-scale", type=float, default=None,
                        help="Multiplier on the LAST horizon-step reward. "
                             "Default 3.0.")
    parser.add_argument("--cube-pos-deadband", type=float, default=0.0,
                        help="Wrist-frame radius (m) inside which cube_pos "
                             "error is zero. Recommended ~0.02 for cube_pitch.")
    parser.add_argument("--cube-pos-wall-start", type=float, default=0.0,
                        help="Distance past the deadband (m) where the "
                             "quadratic wall kicks in. e.g. 0.01 = 10 mm "
                             "linear band before the wall. Default 0.")
    parser.add_argument("--cube-pos-wall-scale", type=float, default=0.0,
                        help="Quadratic penalty weight (per m²) past the "
                             "wall threshold. e.g. 100 makes a 70 mm wall "
                             "excess contribute ~0.49 per step vs 0.07 "
                             "linear. Default 0 (no wall).")
    parser.add_argument("--finger-noise", type=float, default=None)
    parser.add_argument("--wrist-noise", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Kinematic-pin warmup substeps before MPPI starts.")
    parser.add_argument("--warmup-closure", type=float, default=0.0)
    parser.add_argument("--loop-closure", type=float, default=0.0)
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                        help="Cube pos-error threshold (m) for resample-on-drop.")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "mjwp"],
                        help="Rollout backend: cpu or mjwp (NVIDIA Warp GPU).")
    parser.add_argument("--time-scale", type=int, default=1,
                        help="Integer reference-trajectory upsampling.")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Cap MPC at this many ticks. 0 = full sequence.")
    parser.add_argument("--stale-rotation-window", type=float, default=0.0,
                        help="Early-stop if cube rotates less than "
                             "--stale-rotation-deg over this many seconds.")
    parser.add_argument("--stale-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()

    weights = _build_weights(args)
    weights.cube_pos_deadband_m = float(args.cube_pos_deadband)
    weights.cube_pos_wall_start_m = float(args.cube_pos_wall_start)
    weights.cube_pos_wall_scale = float(args.cube_pos_wall_scale)
    cfg = MPPIConfig(
        num_samples=args.num_samples,
        horizon_steps=args.horizon,
        max_iters=args.iters,
        n_substeps=args.substeps,
        weights=weights,
        drop_threshold_m=args.drop_threshold,
    )
    if args.finger_noise is not None: cfg.finger_noise_std = args.finger_noise
    if args.wrist_noise is not None:  cfg.wrist_noise_std = args.wrist_noise

    print(f"  weights: cube_pos={weights.cube_pos}  cube_quat={weights.cube_quat}"
          f"  finger_qpos={weights.finger_qpos}  contact={weights.contact}"
          f"  wrist_delta={weights.wrist_delta}")

    for d in args.session_dirs:
        print(f"\n{d.name}")
        run_session(
            d, force=args.force, cfg=cfg, seed=args.seed,
            warmup_steps=args.warmup_steps,
            warmup_closure=args.warmup_closure,
            loop_closure=args.loop_closure,
            backend=args.backend,
            time_scale=args.time_scale,
            max_ticks=args.max_ticks,
            stale_rotation_window_s=args.stale_rotation_window,
            stale_rotation_deg=args.stale_rotation_deg,
        )


if __name__ == "__main__":
    main()
