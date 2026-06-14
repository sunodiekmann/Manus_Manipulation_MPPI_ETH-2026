"""Stage 3 — Dynamic retargeting via SPIDER-style sampling MPC.

Reads Stage 2 (`retargeted_sequence.npz`) + Stage 2.5 contact targets
(`contact_sequence.npz`) and produces a dynamically feasible trajectory
by rolling MPPI in the base physics scene (cube as a real freejoint,
no virtual stick — fingers must grip and rotate the cube under physics).

Action layout (per step): [20 finger ctrls, 3 wrist axis-angle delta].
Wrist policy: position locked to ref; orientation = ref ⊗ exp(action delta).

Inputs (per session):
    recordings/<session>/retargeted_sequence.npz   (Stage 2: SPIDER equality IK)
    recordings/<session>/contact_sequence.npz      (Stage 2.5: detect_contacts)

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
import json
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

def _build_env(backend: str, num_samples: int, sim_dt: float,
               spider_physics: bool,
               cube_edge_m: float | None = None,
               cube_mass_kg: float | None = None):
    if backend == "cpu":
        return CpuEnv(num_samples=num_samples, sim_dt=sim_dt,
                      spider_physics=spider_physics,
                      cube_edge_m=cube_edge_m, cube_mass_kg=cube_mass_kg)
    if backend == "mjwp":
        from scripts.retargeting.dynamic_retargeting.mjwp_env import MjwpEnv
        return MjwpEnv(num_samples=num_samples, sim_dt=sim_dt,
                       spider_physics=spider_physics,
                       cube_edge_m=cube_edge_m, cube_mass_kg=cube_mass_kg)
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


def _resample_reference_to_dt(
    qpos_ref: np.ndarray,                  # (N, 27)
    cube_ref: np.ndarray,                  # (N, 7) wxyz
    contact_target: np.ndarray | None,     # (N, 5, 3) cube-frame, NaN allowed
    timeline: np.ndarray,                  # (N,) absolute seconds
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Resample the reference onto a uniform `target_dt` grid (in real time).

    This makes one MPC tick = one reference frame = `target_dt` of simulation,
    so control granularity (ctrl_dt = n_substeps·sim_dt) matches SPIDER's
    decoupled timing. Downsamples (coarse control) when target_dt > native
    ref_dt; upsamples otherwise. Quaternions SLERP, everything else linear;
    contact targets stay NaN where either bracketing native frame is NaN.
    """
    from scipy.spatial.transform import Rotation as R, Slerp
    t_in = np.asarray(timeline, dtype=float)
    N = qpos_ref.shape[0]
    if N < 2:
        return qpos_ref, cube_ref, contact_target, timeline
    t0, t1 = float(t_in[0]), float(t_in[-1])
    M = max(2, int(np.floor((t1 - t0) / target_dt)) + 1)
    t_out = t0 + target_dt * np.arange(M)
    t_out = np.clip(t_out, t0, t1)

    new_qpos = np.zeros((M, 27), dtype=np.float64)
    for j in (0, 1, 2, *range(7, 27)):
        new_qpos[:, j] = np.interp(t_out, t_in, qpos_ref[:, j])
    wrist_xyzw = qpos_ref[:, [4, 5, 6, 3]]
    wq = Slerp(t_in, R.from_quat(wrist_xyzw))(t_out).as_quat()
    new_qpos[:, 3] = wq[:, 3]
    new_qpos[:, 4:7] = wq[:, :3]

    new_cube = np.zeros((M, 7), dtype=np.float64)
    for j in range(3):
        new_cube[:, j] = np.interp(t_out, t_in, cube_ref[:, j])
    cube_xyzw = cube_ref[:, [4, 5, 6, 3]]
    cq = Slerp(t_in, R.from_quat(cube_xyzw))(t_out).as_quat()
    new_cube[:, 3] = cq[:, 3]
    new_cube[:, 4:7] = cq[:, :3]

    new_contact = None
    if contact_target is not None:
        F = contact_target.shape[1]
        new_contact = np.full((M, F, 3), np.nan, dtype=np.float64)
        for f in range(F):
            valid = ~np.isnan(contact_target[:, f, 0])
            for c in range(3):
                vals = contact_target[:, f, c]
                interp = np.interp(t_out, t_in, np.where(valid, vals, 0.0))
                # mark NaN where either bracketing native frame was invalid
                left = np.searchsorted(t_in, t_out, side="right") - 1
                left = np.clip(left, 0, N - 2)
                ok = valid[left] & valid[left + 1]
                new_contact[ok, f, c] = interp[ok]
    return new_qpos, new_cube, new_contact, t_out


def run_session(
    session_dir: Path,
    *, force: bool, cfg: MPPIConfig, seed: int,
    warmup_steps: int = 50,
    backend: str = "cpu",
    time_scale: int = 1,
    max_ticks: int = 0,
    tag: str | None = None,
    spider_physics: bool = True,
    print_every: int = 10,
    cube_edge_m: float | None = None,
    cube_mass_kg: float | None = None,
    start_time_s: float = 0.0,
    stage2_tag: str | None = None,
) -> None:
    # Stage 2 / 2.5 outputs — tagged variants if a stage2_tag is supplied
    # (used for per-cube-edge reruns).
    rt_name = (f"retargeted_sequence__{stage2_tag}.npz" if stage2_tag
               else "retargeted_sequence.npz")
    ct_name = (f"contact_sequence__{stage2_tag}.npz" if stage2_tag
               else "contact_sequence.npz")
    retarget_path = session_dir / rt_name
    contact_path = session_dir / ct_name
    out_name = (f"retargeted_sequence_dynamic__{tag}.npz" if tag
                else "retargeted_sequence_dynamic.npz")
    out_path = session_dir / out_name

    if not retarget_path.exists():
        raise FileNotFoundError(
            f"missing {retarget_path.name}. "
            "Run scripts.retargeting.kinematic_retargeting first.")
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

    # Control-rate decoupling (SPIDER-style): one MPC tick = one control step =
    # ctrl_dt = n_substeps · sim_dt. Resample the reference onto that grid so a
    # tick is real-time-correct and control is as coarse as SPIDER's (0.4s),
    # which is the key robustness lever (no twitchy knife-edge solutions).
    ctrl_dt = cfg.n_substeps * cfg.sim_dt
    native_dt = float(np.diff(timeline).mean()) if len(timeline) > 1 else ctrl_dt
    N_native = qpos_ref.shape[0]
    qpos_ref, cube_ref, contact_target_per_frame, timeline = _resample_reference_to_dt(
        qpos_ref, cube_ref, contact_target_per_frame, timeline, ctrl_dt)
    print(f"  ctrl_dt={ctrl_dt:.3f}s (= {cfg.n_substeps}·{cfg.sim_dt}); reference "
          f"resampled {N_native} frames @ {native_dt*1000:.0f}ms → "
          f"{qpos_ref.shape[0]} ctrl steps @ {ctrl_dt*1000:.0f}ms")

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

    # Late-start: skip the first `start_frame` reference ticks so MPPI begins
    # from a (potentially less-stable) mid-motion pose.
    start_frame = max(0, int(round(start_time_s / ctrl_dt))) if start_time_s > 0 else 0
    if start_frame > 0:
        if start_frame >= N:
            raise ValueError(
                f"--start-time {start_time_s}s ({start_frame} frames) >= "
                f"total ref length {N}")
        qpos_ref = qpos_ref[start_frame:]
        cube_ref = cube_ref[start_frame:]
        timeline = timeline[start_frame:]
        if contact_target_per_frame_padded is not None:
            contact_target_per_frame_padded = contact_target_per_frame_padded[start_frame:]
        # Re-pad after trim.
        N = qpos_ref.shape[0]
        qpos_ref_padded = _pad_reference(qpos_ref, N + H)
        cube_ref_padded = _pad_reference(cube_ref, N + H)
        print(f"  late-start: skipping first {start_frame} ref ticks "
              f"({start_time_s:.2f}s) → {N} remaining")

    print(f"  backend: {backend}  spider_physics: {spider_physics}")
    if cube_edge_m is not None or cube_mass_kg is not None:
        print(f"  cube override: "
              f"edge={cube_edge_m if cube_edge_m is not None else 'XML'} m  "
              f"mass={cube_mass_kg if cube_mass_kg is not None else 'XML'} kg")
    env = _build_env(backend, cfg.num_samples, cfg.sim_dt, spider_physics,
                     cube_edge_m=cube_edge_m, cube_mass_kg=cube_mass_kg)
    env.set_world_state(qpos_ref[0], cube_ref[0])

    # Kinematic-pin warmup: pin cube at ref[0], step physics so fingers
    # settle into a grip. After this the cube is released and runs free.
    if warmup_steps > 0:
        env.warmup_grasp(
            cube_pose7=cube_ref[0],
            finger_ctrl=qpos_ref[0, 7:27],
            ref_wrist_pos=qpos_ref[0, 0:3],
            ref_wrist_quat=qpos_ref[0, 3:7],
            n_steps=warmup_steps,
        )
        print(f"  warmup grasp: {warmup_steps} substeps with cube pinned")

    rng = np.random.default_rng(seed)

    qpos_traj = np.zeros((N, 27), dtype=np.float64)
    cube_traj = np.zeros((N, 7), dtype=np.float64)
    ctrl_traj = np.zeros((N, N_FINGER_CTRL), dtype=np.float64)
    wrist_delta_traj = np.zeros((N, N_WRIST_DELTA), dtype=np.float64)
    rew_max_log, rew_mean_log, plan_rew_log = [], [], []
    pos_err_history = np.zeros(N, dtype=np.float64)
    rot_err_history = np.zeros(N, dtype=np.float64)
    iter_improvement_history = np.zeros(N, dtype=np.float64)
    # Per-tick chosen-plan terms (chosen plan = post-MPPI-iter applied control,
    # not best sample, not mean across samples). This is the honest "what did
    # the planner commit to?" signal.
    term_means_history = np.zeros((N, len(TERM_NAMES)), dtype=np.float64)
    plan_rew_history = np.zeros(N, dtype=np.float64)

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
        # Interpolate the applied control toward the next control step (plan[1]
        # + next reference wrist) across the substeps — smooth spline, not a
        # 0.4s staircase.
        nxt = 1 if H > 1 else 0
        env.step_world(
            action=action,
            ref_wrist_pos=ref_wrist_pos_h[0],
            ref_wrist_quat=ref_wrist_quat_h[0],
            n_substeps=cfg.n_substeps,
            next_action=plan[nxt],
            next_ref_wrist_pos=ref_wrist_pos_h[nxt],
            next_ref_wrist_quat=ref_wrist_quat_h[nxt],
            record_stride=0,
        )

        qpos_traj[t] = env.world.qpos[:N_FINGER_CTRL + 7]
        cube_traj[t] = env.get_world_cube_pose7()
        ctrl_traj[t] = action[:N_FINGER_CTRL]
        wrist_delta_traj[t] = action[N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA]
        rew_max_log.append(info["rew_max"])
        rew_mean_log.append(info["rew_mean"])
        plan_rew_log.append(info["plan_rew"])

        sim_cube_pose = cube_traj[t]
        ref_cube_pose_now = cube_ref_padded[t]
        pos_err_history[t] = float(np.linalg.norm(
            sim_cube_pose[0:3] - ref_cube_pose_now[0:3]))
        dot = float(abs(np.dot(sim_cube_pose[3:7], ref_cube_pose_now[3:7])))
        rot_err_history[t] = float(2.0 * np.arccos(min(1.0, max(-1.0, dot))))
        iter_improvement_history[t] = (
            info["rew_max"][-1] - info["rew_max"][0])
        for ti, k in enumerate(TERM_NAMES):
            term_means_history[t, ti] = info["plan_terms"][k][-1]
        plan_rew_history[t] = info["plan_rew"][-1]

        # Slide horizon: drop applied step, append next IK ref tail.
        plan = np.concatenate(
            [plan[1:], np.zeros((1, N_ACTION), dtype=np.float64)], axis=0)
        plan[-1, :N_FINGER_CTRL] = qpos_ref_padded[t + H, 7:27]
        plan[-1, N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0

        if t % max(1, print_every) == 0:
            elapsed = time.perf_counter() - t_start
            rate = (t + 1) / elapsed
            eta = (N_run - t - 1) / rate if rate > 0 else 0.0
            print(f"  t={t:4d}/{N}  realized: pos={pos_err_history[t]*1000:5.1f}mm  "
                  f"rot={np.degrees(rot_err_history[t]):5.1f}°  "
                  f"frac_term={info['frac_terminated'][-1]:.2f}  "
                  f"rate={rate:.2f} t/s  ETA={eta:.0f}s")
            print(f"           plan: rew={info['plan_rew'][-1]:+7.2f}  "
                  f"best_sample={info['rew_max'][-1]:+7.2f}  "
                  f"(MPPI improv {iter_improvement_history[t]:+5.2f})")
            if cfg.eval_plan_each_iter and len(info["plan_rew"]) > 1:
                rmax_row = "  ".join(f"{r:+5.2f}" for r in info["rew_max"])
                prew_row = "  ".join(f"{r:+5.2f}" for r in info["plan_rew"])
                print(f"           per-iter rew_max:  {rmax_row}")
                print(f"           per-iter plan_rew: {prew_row}")
            terms_str = "  ".join(
                f"{k}={info['plan_terms'][k][-1]:+6.2f}" for k in TERM_NAMES)
            print(f"           plan terms (forecast): {terms_str}")

        # Early-stop on full drop: if every sample in the last iter terminated
        # (cube is gone, MPPI has nothing to recover with), end the run here.
        # The partial trajectory up to and including this tick still gets
        # saved — K = len(rew_max_log) below.
        if info["frac_terminated"][-1] >= 1.0:
            print(f"  ✗ cube dropped at t={t}/{N} "
                  f"(frac_term=1.00). Ending run, saving partial trajectory.")
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
    plan_rew_log = plan_rew_log[:K]
    # Save the (post-resample) reference cube + hand that MPC was actually
    # optimizing against. Replay reads these for the ghost overlay so the
    # frame count matches the dynamic trajectory even with --time-scale.
    cube_ref_used = cube_ref_padded[:K].copy()
    qpos_ref_used = qpos_ref_padded[:K].copy()

    w = cfg.weights
    run_config = {
        "tag": tag,
        "backend": backend,
        "seed": seed,
        "num_samples": cfg.num_samples,
        "iters": cfg.max_iters,
        "horizon": cfg.horizon_steps,
        "num_knots": cfg.num_knots,
        "substeps": cfg.n_substeps,
        "sim_dt": cfg.sim_dt,
        "ctrl_dt": cfg.n_substeps * cfg.sim_dt,
        "spider_physics": spider_physics,
        "cube_edge_m": cube_edge_m,
        "cube_mass_kg": cube_mass_kg,
        "start_time_s": start_time_s,
        "start_frame": start_frame,
        "stage2_tag": stage2_tag,
        "time_scale": time_scale,
        "warmup_steps": warmup_steps,
        "finger_noise": cfg.finger_noise_std,
        "wrist_noise": cfg.wrist_noise_std,
        "top_frac": cfg.top_frac,
        "softmax_temp": cfg.softmax_temp,
        "first_step_noise_scale": cfg.first_step_noise_scale,
        "last_step_noise_scale": cfg.last_step_noise_scale,
        "final_noise_scale": cfg.final_noise_scale,
        "exploit_ratio": cfg.exploit_ratio,
        "exploit_noise_scale": cfg.exploit_noise_scale,
        "drop_threshold_m": cfg.drop_threshold_m,
        "drop_threshold_rad": cfg.drop_threshold_rad,
        "improvement_threshold": cfg.improvement_threshold,
        "terminal_scale": w.terminal_scale,
        "weights": {
            "cube_pos": w.cube_pos, "cube_quat": w.cube_quat,
            "finger_qpos": w.finger_qpos, "contact": w.contact,
            "wrist_delta": w.wrist_delta, "qvel": w.qvel,
            "cube_angvel": w.cube_angvel,
        },
        "wall_time_s": round(elapsed, 1),
        "ticks_per_s": round(len(timeline) / elapsed, 3) if elapsed > 0 else 0.0,
    }
    # Per-iter logs may have varying length per tick now that early stopping
    # is on. Pad to max_iters with NaN so we get a clean (K, max_iters) array
    # in the npz; consumers can mask NaN to recover the actual iter count.
    def _pad_iters(lst_of_lst, max_iters):
        arr = np.full((len(lst_of_lst), max_iters), np.nan, dtype=np.float64)
        for i, sub in enumerate(lst_of_lst):
            n = min(len(sub), max_iters)
            arr[i, :n] = sub[:n]
        return arr
    save_kwargs = dict(
        timeline=timeline,
        run_config=np.array(json.dumps(run_config)),
        retargeted_qpos=qpos_traj,
        cube_traj=cube_traj,
        ctrl_traj=ctrl_traj,
        wrist_delta_traj=wrist_delta_traj,
        rew_max=_pad_iters(rew_max_log, cfg.max_iters),
        rew_mean=_pad_iters(rew_mean_log, cfg.max_iters),
        pos_err_history=pos_err_history[:K],
        rot_err_history=rot_err_history[:K],
        iter_improvement_history=iter_improvement_history[:K],
        term_names=np.asarray(list(TERM_NAMES)),
        term_means_history=term_means_history[:K],  # NOTE: now chosen-plan terms (was sample mean)
        plan_rew_history=plan_rew_history[:K],
        plan_rew_per_iter=_pad_iters(plan_rew_log, cfg.max_iters),  # (K, max_iters); NaN where early-stopped
        cube_ref_used=cube_ref_used,
        qpos_ref_used=qpos_ref_used,
    )
    np.savez_compressed(out_path, **save_kwargs)
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
    """Start from the validated defaults (RewardWeights), apply any explicit
    --w-* overrides on top."""
    w = RewardWeights()
    for field in ("cube_pos", "cube_quat", "finger_qpos", "contact",
                  "wrist_delta", "qvel", "cube_angvel"):
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
    parser.add_argument("--num-samples", type=int, default=1024,
                        help="MPPI samples per iter (SPIDER-matched 1024).")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Control steps in the planning horizon "
                             "(SPIDER horizon/ctrl_dt = 1.6/0.4 = 4).")
    parser.add_argument("--iters", type=int, default=16,
                        help="MPPI refinement iters per step (SPIDER 16).")
    parser.add_argument("--num-knots", type=int, default=None,
                        help="Control knots interpolated to horizon. None = "
                             "use MPPIConfig default. Set equal to --horizon "
                             "for no interpolation (each horizon step has its "
                             "own free control).")
    parser.add_argument("--substeps", type=int, default=40,
                        help="mj_step per control step. ctrl_dt = substeps·sim_dt "
                             "(SPIDER 40·0.01 = 0.4s). The reference is resampled "
                             "to this ctrl_dt.")
    parser.add_argument("--sim-dt", type=float, default=0.005,
                        help="Physics timestep. SPIDER 0.01; we use 0.005 "
                             "because stiff solimp needs finer integration.")
    parser.add_argument("--no-spider-physics", dest="spider_physics",
                        action="store_false",
                        help="Disable SPIDER soft-contact + condim-4 physics "
                             "(revert to the original hard contacts) for A/B.")
    parser.add_argument("--w-cube-pos", type=float, default=None)
    parser.add_argument("--w-cube-quat", type=float, default=None)
    parser.add_argument("--w-finger-qpos", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--w-wrist-delta", type=float, default=None)
    parser.add_argument("--w-qvel", type=float, default=None,
                        help="Velocity penalty weight (per rad/s, m/s). "
                             "Encourages stable equilibrium states. Default 0.005.")
    parser.add_argument("--w-cube-angvel", type=float, default=None,
                        help="Cube angular-velocity (per-tick rotation "
                             "increment) tracking weight. Default 0 (pure "
                             "pose tracking). Raise it for motion/task-style "
                             "tracking; pair with --w-cube-quat 0 to fully "
                             "switch from absolute-pose to motion tracking "
                             "(cube_pos still anchors position).")
    parser.add_argument("--terminal-scale", type=float, default=None,
                        help="Multiplier on the LAST horizon-step reward. "
                             "Default 3.0.")
    parser.add_argument("--finger-noise", type=float, default=None)
    parser.add_argument("--wrist-noise", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Kinematic-pin warmup substeps before MPPI starts.")
    parser.add_argument("--cube-edge", type=float, default=None,
                        metavar="M",
                        help="Override cube edge length (m). Default = XML "
                             "(0.07). Rescales geom_size and recomputes "
                             "inertia for a uniform solid cube. NOTE: contact "
                             "targets are cube-frame and assume the XML edge "
                             "length — for honest comparison, re-run Stage 2 "
                             "(kinematic_retargeting) with the new edge "
                             "and pass --stage2-tag to point at the rerun.")
    parser.add_argument("--cube-mass", type=float, default=None,
                        metavar="KG",
                        help="Override cube mass (kg). Default = XML "
                             "(0.04). Rescales body_inertia linearly. "
                             "No Stage 2 rerun needed (kinematic-only).")
    parser.add_argument("--start-time", type=float, default=0.0,
                        metavar="S",
                        help="Late-start: skip the first SECONDS of the "
                             "reference (MPPI begins from a mid-motion pose). "
                             "Backed by ctrl_dt — start_frame = round(S/ctrl_dt).")
    parser.add_argument("--stage2-tag", type=str, default=None,
                        help="Load tagged Stage 2 outputs: "
                             "retargeted_sequence__<TAG>.npz / "
                             "contact_sequence__<TAG>.npz. Use after a "
                             "Stage 2 re-run for a different cube edge.")
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                        help="Cube pos-error threshold (m) for resample-on-drop.")
    parser.add_argument("--drop-threshold-deg", type=float, default=None,
                        help="Cube rot-error threshold (deg) for resample-on-drop. "
                             "Default 60° (set in MPPIConfig); pass to override.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Diagnostic mode: print every tick + per-iter "
                             "plan-rew evaluation (reveals averaging collapse). "
                             "Roughly doubles wall time.")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "mjwp"],
                        help="Rollout backend: cpu or mjwp (NVIDIA Warp GPU).")
    parser.add_argument("--time-scale", type=int, default=1,
                        help="Integer reference-trajectory upsampling.")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Cap MPC at this many ticks. 0 = full sequence.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for the output npz: writes "
                             "retargeted_sequence_dynamic__<tag>.npz instead "
                             "of overwriting the canonical file. Use to keep "
                             "sweep runs side by side.")
    args = parser.parse_args()

    weights = _build_weights(args)
    cfg = MPPIConfig(
        num_samples=args.num_samples,
        horizon_steps=args.horizon,
        max_iters=args.iters,
        n_substeps=args.substeps,
        sim_dt=args.sim_dt,
        weights=weights,
        drop_threshold_m=args.drop_threshold,
    )
    if args.num_knots is not None:    cfg.num_knots = args.num_knots
    if args.finger_noise is not None: cfg.finger_noise_std = args.finger_noise
    if args.wrist_noise is not None:  cfg.wrist_noise_std = args.wrist_noise
    if args.drop_threshold_deg is not None:
        cfg.drop_threshold_rad = float(np.radians(args.drop_threshold_deg))
    cfg.eval_plan_each_iter = bool(args.diagnostics)
    print_every = 1 if args.diagnostics else 10

    print(f"  weights: cube_pos={weights.cube_pos}  cube_quat={weights.cube_quat}"
          f"  finger_qpos={weights.finger_qpos}  contact={weights.contact}"
          f"  wrist_delta={weights.wrist_delta}  qvel={weights.qvel}"
          f"  cube_angvel={weights.cube_angvel}")

    for d in args.session_dirs:
        print(f"\n{d.name}")
        run_session(
            d, force=args.force, cfg=cfg, seed=args.seed,
            warmup_steps=args.warmup_steps,
            backend=args.backend,
            time_scale=args.time_scale,
            max_ticks=args.max_ticks,
            tag=args.tag,
            spider_physics=args.spider_physics,
            print_every=print_every,
            cube_edge_m=args.cube_edge,
            cube_mass_kg=args.cube_mass,
            start_time_s=args.start_time,
            stage2_tag=args.stage2_tag,
        )


if __name__ == "__main__":
    main()
