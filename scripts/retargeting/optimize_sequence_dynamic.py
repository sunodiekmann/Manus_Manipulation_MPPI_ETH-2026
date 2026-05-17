"""Stage 3 — Dynamic retargeting via SPIDER-style sampling MPC.

Reads Stage 2 (`retargeted_sequence.npz`) + Stage 2.5 contact targets
(`contact_sequence.npz`) and produces a dynamically feasible trajectory by
rolling MPPI in the *dyn* physics scene from `build_ik_scene.compile_dyn`:
cube freejoint replaced by 3 slide + 3 hinge joints + position actuators
("virtual stick"), so MPC can supply gradient-rich object commands that
fade as the schedule progresses and fingers learn to hold the cube alone.

Action layout (per step):
    [20 finger ctrls, 3 wrist axis-angle delta, 3 obj_pos ctrl, 3 obj_rot ctrl]

Wrist policy: position locked to ref; orientation = ref ⊗ exp(action delta).

Inputs (per session):
    recordings/<session>/retargeted_sequence.npz   (Stage 2: SPIDER equality IK)
    recordings/<session>/contact_sequence.npz      (Stage 2.5: spider_contact)

Output:
    recordings/<session>/retargeted_sequence_dynamic.npz
        timeline (N,)
        retargeted_qpos (N, 27)        — sim hand qpos after each MPC tick
        cube_traj (N, 7)               — sim cube pose7 (pos+wxyz)
        ctrl_traj (N, 20)              — applied finger controls
        wrist_delta_traj (N, 3)        — applied wrist axis-angle delta
        obj_ctrl_traj (N, 6)           — applied obj pos+rot ctrls
        obj_decay_traj (N,)            — virtual-stick gain factor each tick
        rew_max, rew_mean              — MPPI diagnostics per tick

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
    N_OBJ_CTRL,
    N_OBJ_POS_CTRL,
    N_WRIST_DELTA,
    cube_qpos_from_pose7,
)
from scripts.retargeting.dynamic_retargeting.mppi import MPPIConfig, optimize_step
from scripts.retargeting.dynamic_retargeting.reward import RewardWeights


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


def _build_obj_baseline(cube_ref_pose7: np.ndarray) -> np.ndarray:
    """(N, 7) ref cube pose7 → (N, 6) cube qpos in dyn-scene coords (Euler XYZ)."""
    out = np.zeros((cube_ref_pose7.shape[0], N_OBJ_CTRL), dtype=np.float64)
    for i in range(cube_ref_pose7.shape[0]):
        out[i] = cube_qpos_from_pose7(cube_ref_pose7[i])
    return out




def _shift_cube_reference(cube_ref: np.ndarray, wrist_T: np.ndarray,
                          shift_wrist: np.ndarray) -> np.ndarray:
    """Translate the cube reference by `shift_wrist` (wrist-frame, metres)
    at every frame. Keeps cube orientation untouched. Used to push the
    cube away from the palm so MPPI doesn't have to fight a geometry where
    the reference cube grazes the hand interior (cube_pitch case).

    `wrist_T` is the (N, 27) hand qpos; first 7 entries are the freejoint
    pose (pos + wxyz quat). Result is a copy.
    """
    if np.allclose(shift_wrist, 0.0):
        return cube_ref
    out = cube_ref.copy()
    from scipy.spatial.transform import Rotation as R
    for i in range(cube_ref.shape[0]):
        wrist_quat_wxyz = wrist_T[i, 3:7]
        wrist_quat_xyzw = np.array([wrist_quat_wxyz[1], wrist_quat_wxyz[2],
                                    wrist_quat_wxyz[3], wrist_quat_wxyz[0]])
        R_world_wrist = R.from_quat(wrist_quat_xyzw).as_matrix()
        out[i, :3] = cube_ref[i, :3] + R_world_wrist @ shift_wrist
    return out


def run_session(
    session_dir: Path,
    *, force: bool, cfg: MPPIConfig, seed: int,
    warmup_steps: int = 50, warmup_closure: float = 0.0,
    loop_closure: float = 0.0,
    backend: str = "cpu",
    cube_shift: np.ndarray | None = None,
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

    if cube_shift is not None and not np.allclose(cube_shift, 0.0):
        cube_ref = _shift_cube_reference(cube_ref, qpos_ref, cube_shift)
        print(f"  cube_shift (wrist frame, mm): "
              f"[{cube_shift[0]*1000:+.1f}, {cube_shift[1]*1000:+.1f}, {cube_shift[2]*1000:+.1f}]")
    cm = np.load(contact_path)
    contact_target_per_frame = (
        np.asarray(cm["contact_target_per_frame"], dtype=float)
        if "contact_target_per_frame" in cm.files else None
    )

    N = qpos_ref.shape[0]
    H = cfg.horizon_steps
    qpos_ref_padded = _pad_reference(qpos_ref, N + H)
    cube_ref_padded = _pad_reference(cube_ref, N + H)
    obj_baseline_padded = _build_obj_baseline(cube_ref_padded)       # (N+H, 6)
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
    print(f"  noise std: finger={cfg.finger_noise_std}  wrist={cfg.wrist_noise_std}"
          f"  obj_pos={cfg.obj_pos_noise_std}  obj_rot={cfg.obj_rot_noise_std}")
    print(f"  virtual stick: iter_start={cfg.iter_stick_start}  iter_end={cfg.iter_stick_end}  "
          f"(annealed across {cfg.max_iters} MPPI iters; world step always stick=0)")

    print(f"  backend: {backend}")
    env = _build_env(backend, cfg.num_samples)
    env.set_world_state(qpos_ref[0], cube_ref[0])
    if loop_closure > 0.0:
        env.set_closure_bias(loop_closure)

    # Virtual-stick warmup: stick is fully on, fingers settle into a grip
    # before MPC takes over.
    env.set_obj_decay(1.0)
    if warmup_steps > 0:
        warmup_action = np.zeros(N_ACTION, dtype=np.float64)
        warmup_action[:N_FINGER_CTRL] = qpos_ref[0, 7:27]
        if warmup_closure > 0.0:
            warmup_action[:N_FINGER_CTRL] += (
                warmup_closure * env._flexion_closure_direction())
        warmup_action[N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0
        warmup_action[N_FINGER_CTRL + N_WRIST_DELTA:] = obj_baseline_padded[0]
        for _ in range(warmup_steps):
            env.step_world(
                action=warmup_action,
                ref_wrist_pos=qpos_ref[0, 0:3],
                ref_wrist_quat=qpos_ref[0, 3:7],
                n_substeps=1,
            )
        print(f"  warmup grasp: {warmup_steps} substeps with stick=1.0  "
              f"closure_bias={warmup_closure:.3f} rad")

    rng = np.random.default_rng(seed)

    qpos_traj = np.zeros((N, 27), dtype=np.float64)
    cube_traj = np.zeros((N, 7), dtype=np.float64)
    ctrl_traj = np.zeros((N, N_FINGER_CTRL), dtype=np.float64)
    wrist_delta_traj = np.zeros((N, N_WRIST_DELTA), dtype=np.float64)
    obj_ctrl_traj = np.zeros((N, N_OBJ_CTRL), dtype=np.float64)
    obj_decay_traj = np.zeros(N, dtype=np.float64)  # always 0 now (kept for back-compat)
    rew_max_log, rew_mean_log = [], []

    # Initialise the plan: finger ctrls = IK ref, wrist delta = 0, obj ctrl = baseline.
    plan = np.zeros((H, N_ACTION), dtype=np.float64)
    plan[:, :N_FINGER_CTRL] = qpos_ref_padded[:H, 7:27]
    plan[:, N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0
    plan[:, N_FINGER_CTRL + N_WRIST_DELTA:] = obj_baseline_padded[:H]

    t_start = time.perf_counter()
    for t in range(N):
        ref_qpos_h = qpos_ref_padded[t : t + H]
        ref_cube_pos_h = cube_ref_padded[t : t + H, 0:3]
        ref_cube_quat_h = cube_ref_padded[t : t + H, 3:7]
        ref_wrist_pos_h = qpos_ref_padded[t : t + H, 0:3]
        ref_wrist_quat_h = qpos_ref_padded[t : t + H, 3:7]
        obj_baseline_h = obj_baseline_padded[t : t + H]
        contact_target_h = (
            contact_target_per_frame_padded[t : t + H]
            if contact_target_per_frame_padded is not None else None
        )

        plan, info = optimize_step(
            env=env,
            plan=plan,
            obj_baseline=obj_baseline_h,
            ref_qpos=ref_qpos_h,
            ref_cube_pos=ref_cube_pos_h,
            ref_cube_quat=ref_cube_quat_h,
            ref_wrist_pos=ref_wrist_pos_h,
            ref_wrist_quat=ref_wrist_quat_h,
            contact_target_per_step=contact_target_h,
            cfg=cfg,
            rng=rng,
        )

        # World step always with stick OFF — the stick is an optimizer
        # warm-start only, not real physics aid.
        env.set_obj_decay(0.0)
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
        obj_ctrl_traj[t] = action[N_FINGER_CTRL + N_WRIST_DELTA:]
        obj_decay_traj[t] = 0.0
        rew_max_log.append(info["rew_max"])
        rew_mean_log.append(info["rew_mean"])

        # Slide horizon: drop applied step, append next IK ref + obj baseline tail.
        plan = np.concatenate(
            [plan[1:], np.zeros((1, N_ACTION), dtype=np.float64)], axis=0)
        plan[-1, :N_FINGER_CTRL] = qpos_ref_padded[t + H, 7:27]
        plan[-1, N_FINGER_CTRL : N_FINGER_CTRL + N_WRIST_DELTA] = 0.0
        plan[-1, N_FINGER_CTRL + N_WRIST_DELTA:] = obj_baseline_padded[t + H]

        if t % 10 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (t + 1) / elapsed
            eta = (N - t - 1) / rate if rate > 0 else 0.0
            print(f"  t={t:4d}/{N}  "
                  f"rew[max]={info['rew_max'][-1]:+8.2f} "
                  f"rew[mean]={info['rew_mean'][-1]:+8.2f}  "
                  f"frac_term[final]={info['frac_terminated'][-1]:.2f}  "
                  f"rate={rate:.2f} ticks/s  ETA={eta:.0f}s")

        # Stale-rotation early stop: if the *sim* cube has rotated less than
        # `stale_rotation_deg` over the last `stale_rotation_window_s` of
        # ticks AND the *reference* is still rotating, the MPC is stuck and
        # further compute is wasted. Disabled when window_s == 0.
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
                    obj_ctrl_traj = obj_ctrl_traj[: t + 1]
                    obj_decay_traj = obj_decay_traj[: t + 1]
                    timeline = timeline[: t + 1]
                    break

    elapsed = time.perf_counter() - t_start
    print(f"  Done in {elapsed:.1f}s ({len(timeline)/elapsed:.2f} ticks/s).")

    rew_max_log = rew_max_log[: len(timeline)]
    rew_mean_log = rew_mean_log[: len(timeline)]
    np.savez_compressed(
        out_path,
        timeline=timeline,
        retargeted_qpos=qpos_traj,
        cube_traj=cube_traj,
        ctrl_traj=ctrl_traj,
        wrist_delta_traj=wrist_delta_traj,
        obj_ctrl_traj=obj_ctrl_traj,
        obj_decay_traj=obj_decay_traj,
        rew_max=np.asarray(rew_max_log),
        rew_mean=np.asarray(rew_mean_log),
    )
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
                             "Encourages stable equilibrium states. Default 0.05.")
    parser.add_argument("--terminal-scale", type=float, default=None,
                        help="Multiplier on the LAST horizon-step reward. "
                             "Forces MPC to care about where the cube ends up "
                             "at horizon-end (fixes myopic drift). Default 10.")
    parser.add_argument("--cube-pos-deadband", type=float, default=0.0,
                        help="Wrist-frame radius (m) inside which cube_pos "
                             "error is zero. Lets MPPI choose any cube pose "
                             "within ±this from the wrist-relative reference, "
                             "then pins it there. Recommended ~0.02 for "
                             "cube_pitch.")
    parser.add_argument("--finger-noise", type=float, default=None)
    parser.add_argument("--wrist-noise", type=float, default=None)
    parser.add_argument("--obj-pos-noise", type=float, default=None)
    parser.add_argument("--obj-rot-noise", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Virtual-stick warmup substeps before MPPI starts.")
    parser.add_argument("--warmup-closure", type=float, default=0.0)
    parser.add_argument("--loop-closure", type=float, default=0.0)
    parser.add_argument("--iter-stick-start", type=float, default=1.0,
                        help="Virtual-stick gain at MPPI iter 0 (warm-start). "
                             "1.0 = full support, 0.0 = no stick.")
    parser.add_argument("--iter-stick-end", type=float, default=0.0,
                        help="Virtual-stick gain at MPPI converged iter. "
                             "Should be 0.0 unless you want a residual stick.")
    parser.add_argument("--iter-stick-schedule", default="warm_then_off",
                        choices=["linear", "warm_then_off"],
                        help="`warm_then_off`: iter 0 = start, iter >=1 = end "
                             "(most iters are pure refinement). `linear`: "
                             "linearly anneal start->end across all iters.")
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                        help="Cube pos-error threshold (m) for resample-on-drop.")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "mjwp"],
                        help="Rollout backend: cpu (sequential CPU MuJoCo) or "
                             "mjwp (NVIDIA Warp GPU, supports large N).")
    parser.add_argument("--cube-shift", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Shift the cube reference (and starting pose) by "
                             "this 3-vector in wrist frame, metres. Use to give "
                             "the cube room from the palm so MPPI doesn't have "
                             "to fight a tight geometry. Doesn't affect "
                             "fingertip contact targets (cube-frame).")
    parser.add_argument("--stale-rotation-window", type=float, default=0.0,
                        help="Early-stop if cube rotates less than "
                             "--stale-rotation-deg over this many seconds while "
                             "the reference is still rotating. 0 disables.")
    parser.add_argument("--stale-rotation-deg", type=float, default=5.0,
                        help="Threshold (deg) for stale-rotation check.")
    args = parser.parse_args()

    weights = _build_weights(args)
    weights.cube_pos_deadband_m = float(args.cube_pos_deadband)
    cfg = MPPIConfig(
        num_samples=args.num_samples,
        horizon_steps=args.horizon,
        max_iters=args.iters,
        n_substeps=args.substeps,
        weights=weights,
        drop_threshold_m=args.drop_threshold,
        iter_stick_start=args.iter_stick_start,
        iter_stick_end=args.iter_stick_end,
        iter_stick_schedule=args.iter_stick_schedule,
    )
    if args.finger_noise is not None: cfg.finger_noise_std = args.finger_noise
    if args.wrist_noise is not None:  cfg.wrist_noise_std = args.wrist_noise
    if args.obj_pos_noise is not None: cfg.obj_pos_noise_std = args.obj_pos_noise
    if args.obj_rot_noise is not None: cfg.obj_rot_noise_std = args.obj_rot_noise

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
            cube_shift=(np.asarray(args.cube_shift, dtype=float)
                        if args.cube_shift is not None else None),
            stale_rotation_window_s=args.stale_rotation_window,
            stale_rotation_deg=args.stale_rotation_deg,
        )


if __name__ == "__main__":
    main()
