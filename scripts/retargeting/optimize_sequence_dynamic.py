"""Stage 3 — Dynamic retargeting via DIAL-MPC sampling MPC.

Takes a kinematically retargeted sequence (Stage 2) plus the contact mask
and per-finger contact targets (Stage 2.5, spider_contact.py) and produces
a dynamically feasible trajectory by running receding-horizon sampling MPC
in the physics scene.

Wrist policy
------------
Position is locked to the kinematic reference (mocap_pos = ref_wrist_pos).
Orientation is reference ⊗ exp(sampled axis-angle delta), so MPC can apply
small wrist rotations to keep the cube on track.

Action layout (per MPC tick): [20 finger ctrls, 3 wrist axis-angle delta].

Inputs (per session):
    recordings/<session>/retargeted_sequence_cube_relative.npz
    recordings/<session>/contact_sequence.npz

Output:
    recordings/<session>/retargeted_sequence_dynamic.npz
        timeline (N,)
        retargeted_qpos (N, 27)        — sim hand qpos after each MPC tick
        ctrl_traj (N, 20)              — applied finger controls
        wrist_delta_traj (N, 3)        — applied wrist axis-angle delta
        cube_traj (N, 7)               — sim cube freejoint pose
        info: aggregated optimisation diagnostics

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
)
from scripts.retargeting.dynamic_retargeting.mp_env import MpCpuEnv
from scripts.retargeting.dynamic_retargeting.mppi import (
    MPPIConfig,
    make_cpu_score_fn,
    optimize_step,
)
from scripts.retargeting.dynamic_retargeting.reward import RewardWeights

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"


def _pad_reference(arr: np.ndarray, total_len: int) -> np.ndarray:
    """Repeat the last frame to extend reference for the trailing horizon."""
    if arr.shape[0] >= total_len:
        return arr[:total_len]
    pad = np.repeat(arr[-1:], total_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def run_session(session_dir: Path, force: bool, cfg: MPPIConfig, seed: int,
                num_workers: int, warmup_steps: int = 50,
                warmup_closure: float = 0.0,
                loop_closure: float = 0.0,
                drop_threshold_m: float = 0.10) -> None:
    feasible_path = session_dir / "retargeted_sequence_feasible.npz"
    cube_rel_path = session_dir / "retargeted_sequence_cube_relative.npz"
    contact_path = session_dir / "contact_sequence.npz"
    out_path = session_dir / "retargeted_sequence_dynamic.npz"

    # Prefer Stage 2c feasibility-IK output; fall back to Stage 2a cube-relative.
    if feasible_path.exists():
        retarget_path = feasible_path
        cube_key = "cube_pose_smoothed"
        print(f"  using Stage 2c plan: {retarget_path.name}")
    elif cube_rel_path.exists():
        retarget_path = cube_rel_path
        cube_key = "cube_pose_used"
        print(f"  using Stage 2a plan: {retarget_path.name}  "
              f"(Stage 2c output not found)")
    else:
        raise FileNotFoundError(
            f"missing both {feasible_path.name} and {cube_rel_path.name}"
        )
    if not contact_path.exists():
        raise FileNotFoundError(f"missing {contact_path}")
    if out_path.exists() and not force:
        print(f"  [skip] {out_path.name} already exists (--force to overwrite)")
        return

    rt = np.load(retarget_path)
    timeline = np.asarray(rt["timeline"], dtype=float)
    qpos_ref = np.asarray(rt["retargeted_qpos"], dtype=float)        # (N, 27)
    cube_ref = np.asarray(rt[cube_key], dtype=float)                 # (N, 7)
    cm = np.load(contact_path)
    contact_mask_seq = np.asarray(cm["contact_mask"], dtype=bool)    # (N, 5)
    contact_target_cube = np.asarray(cm["contact_target_cube"], dtype=float)  # (5, 3)
    contact_target_valid = np.asarray(cm["contact_target_valid"], dtype=bool) # (5,)
    # Per-frame contact targets (segmented). NaN where no target this frame.
    contact_target_per_frame = (
        np.asarray(cm["contact_target_per_frame"], dtype=float)
        if "contact_target_per_frame" in cm.files else None
    )

    N = qpos_ref.shape[0]
    H = cfg.horizon_steps
    qpos_ref_padded = _pad_reference(qpos_ref, N + H)
    cube_ref_padded = _pad_reference(cube_ref, N + H)
    contact_mask_padded = _pad_reference(contact_mask_seq.astype(np.uint8), N + H).astype(bool)
    if contact_target_per_frame is not None:
        # Pad with NaN-bearing copies of the last frame so the trailing horizon
        # has no target (better than reusing the last contact location after
        # the recording ends).
        last_frame = contact_target_per_frame[-1:].copy()
        last_frame[:] = np.nan
        pad = np.repeat(last_frame, (N + H) - contact_target_per_frame.shape[0], axis=0)
        contact_target_per_frame_padded = np.concatenate(
            [contact_target_per_frame, pad], axis=0,
        )
    else:
        contact_target_per_frame_padded = None

    print(f"  sequence length: {N} ref ticks  ({timeline[-1]-timeline[0]:.1f}s)")
    print(f"  MPPI: N={cfg.num_samples}, H={cfg.horizon_steps}, iters={cfg.max_iters},"
          f" substeps={cfg.n_substeps}")
    print(f"  per-finger valid: {dict(zip(['thumb','index','middle','ring','little'], contact_target_valid.tolist()))}")

    # World: single CPU MuJoCo env stepped between MPC ticks.
    world_env = CpuEnv(PHYSICS_XML, num_samples=0)
    world_env.set_world_state(qpos_ref[0], cube_ref[0])
    # Apply loop closure bias to world + rollouts so position-controlled fingers
    # always squeeze toward flexion. MPPI optimises around this baseline.
    if loop_closure > 0.0:
        world_env.set_closure_bias(loop_closure)

    # Virtual-stick warmup grasp (SPIDER §2.3): pin the cube at its starting
    # pose and let fingers settle into contact with the IK plan's frame-0
    # ctrl, so MPPI starts from a stable multi-finger grip rather than
    # whatever transient state mj_forward(qpos_ref[0]) happens to produce.
    if warmup_steps > 0:
        world_env.warmup_grasp(
            cube_pose=cube_ref[0],
            finger_ctrl=qpos_ref[0, 7:27],
            ref_wrist_pos=qpos_ref[0, 0:3],
            ref_wrist_quat=qpos_ref[0, 3:7],
            n_steps=warmup_steps,
            closure_bias=warmup_closure,
        )
        print(f"  warmup grasp: {warmup_steps} substeps with cube pinned"
              f"  closure_bias={warmup_closure:.3f} rad")

    # Batch rollouts: sequential (workers=1) or pool (workers>1).
    use_mp = num_workers > 1
    seq_env = None
    mp_env = None
    if use_mp:
        print(f"  spinning up multiprocessing pool ({num_workers} workers)...")
        mp_env = MpCpuEnv(PHYSICS_XML, num_workers=num_workers)
    else:
        seq_env = CpuEnv(PHYSICS_XML, num_samples=cfg.num_samples)

    # Initial plan: hold-step IK reference for fingers, zero wrist delta.
    plan = np.zeros((H, N_ACTION), dtype=np.float64)
    plan[:, :N_FINGER_CTRL] = qpos_ref_padded[:H, 7:27]

    rng = np.random.default_rng(seed)

    qpos_traj = np.zeros((N, 27), dtype=np.float64)
    cube_traj = np.zeros((N, 7), dtype=np.float64)
    ctrl_traj = np.zeros((N, N_FINGER_CTRL), dtype=np.float64)
    wrist_delta_traj = np.zeros((N, 3), dtype=np.float64)
    rew_max_log = []
    rew_mean_log = []

    t_start = time.perf_counter()
    for t in range(N):
        # reference window
        ref_qpos_h = qpos_ref_padded[t : t + H]
        ref_cube_pos_h = cube_ref_padded[t : t + H, 0:3]
        ref_cube_quat_h = cube_ref_padded[t : t + H, 3:7]
        ref_wrist_pos_h = qpos_ref_padded[t : t + H, 0:3]
        ref_wrist_quat_h = qpos_ref_padded[t : t + H, 3:7]
        contact_mask_t = contact_mask_padded[t]
        contact_target_h = (
            contact_target_per_frame_padded[t : t + H]
            if contact_target_per_frame_padded is not None else None
        )

        if use_mp:
            qpos_init = world_env.world.qpos.copy()
            qvel_init = world_env.world.qvel.copy()
            mocap_pos_init = world_env.world.mocap_pos.copy()
            mocap_quat_init = world_env.world.mocap_quat.copy()
            ctrl_bias = world_env.ctrl_bias.copy()
            def score_fn(samples,
                         _mp=mp_env,
                         _qp=qpos_init, _qv=qvel_init,
                         _mp_pos=mocap_pos_init, _mp_quat=mocap_quat_init,
                         _bias=ctrl_bias,
                         _ct=contact_target_h):
                return _mp.score_samples(
                    samples=samples,
                    ref_qpos=ref_qpos_h,
                    ref_cube_pos=ref_cube_pos_h,
                    ref_cube_quat=ref_cube_quat_h,
                    ref_wrist_pos=ref_wrist_pos_h,
                    ref_wrist_quat=ref_wrist_quat_h,
                    contact_mask=contact_mask_t,
                    contact_target_cube=contact_target_cube,
                    contact_target_valid=contact_target_valid,
                    init_qpos=_qp,
                    init_qvel=_qv,
                    init_mocap_pos=_mp_pos,
                    init_mocap_quat=_mp_quat,
                    n_substeps=cfg.n_substeps,
                    weights=cfg.weights,
                    ctrl_bias=_bias,
                    contact_target_per_step=_ct,
                )
        else:
            score_fn = make_cpu_score_fn(
                seq_env, cfg,
                ref_qpos=ref_qpos_h,
                ref_cube_pos=ref_cube_pos_h,
                ref_cube_quat=ref_cube_quat_h,
                ref_wrist_pos=ref_wrist_pos_h,
                ref_wrist_quat=ref_wrist_quat_h,
                contact_mask=contact_mask_t,
                contact_target_cube=contact_target_cube,
                contact_target_valid=contact_target_valid,
                contact_target_per_step=contact_target_h,
            )
            # Sequential CpuEnv needs world snapshotted into the batch.
            seq_env.world.qpos[:] = world_env.world.qpos
            seq_env.world.qvel[:] = world_env.world.qvel
            seq_env.world.mocap_pos[:] = world_env.world.mocap_pos
            seq_env.world.mocap_quat[:] = world_env.world.mocap_quat

        plan, info = optimize_step(score_fn, plan, cfg, rng)

        # Apply plan[0] to the world env, then slide.
        action = plan[0]
        world_env.step_world(
            action=action,
            ref_wrist_pos=ref_wrist_pos_h[0],
            ref_wrist_quat=ref_wrist_quat_h[0],
            n_substeps=cfg.n_substeps,
        )

        # log applied state
        qpos_traj[t] = world_env.world.qpos[:27]
        cube_traj[t] = world_env.world.qpos[27:34]
        ctrl_traj[t] = action[:N_FINGER_CTRL]
        wrist_delta_traj[t] = action[N_FINGER_CTRL:]
        rew_max_log.append(info["rew_max"])
        rew_mean_log.append(info["rew_mean"])

        # slide horizon: drop applied step, append IK ref for the new tail.
        plan = np.concatenate(
            [plan[1:], np.zeros((1, N_ACTION), dtype=np.float64)],
            axis=0,
        )
        plan[-1, :N_FINGER_CTRL] = qpos_ref_padded[t + H, 7:27]

        if t % 10 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (t + 1) / elapsed
            eta = (N - t - 1) / rate if rate > 0 else 0.0
            print(
                f"  t={t:4d}/{N}  rew[max]={info['rew_max'][-1]:+8.2f} "
                f"rew[mean]={info['rew_mean'][-1]:+8.2f}  rate={rate:.2f} ticks/s  ETA={eta:.0f}s"
            )

        # Early termination: cube has fallen out of the hand (no recovery from here).
        cube_pos_err = float(
            np.linalg.norm(world_env.world.qpos[27:30] - cube_ref[t, 0:3])
        )
        if cube_pos_err > drop_threshold_m:
            print(f"  [terminated] cube dropped at t={t}  pos_err={cube_pos_err*1000:.0f} mm "
                  f"(threshold {drop_threshold_m*1000:.0f} mm)")
            # Trim arrays to written length so the npz reflects what actually ran.
            qpos_traj = qpos_traj[: t + 1]
            cube_traj = cube_traj[: t + 1]
            ctrl_traj = ctrl_traj[: t + 1]
            wrist_delta_traj = wrist_delta_traj[: t + 1]
            timeline = timeline[: t + 1]
            break

    elapsed = time.perf_counter() - t_start
    print(f"  Done in {elapsed:.1f}s ({N/elapsed:.2f} ticks/s).")

    np.savez_compressed(
        out_path,
        timeline=timeline,
        retargeted_qpos=qpos_traj,
        cube_traj=cube_traj,
        ctrl_traj=ctrl_traj,
        wrist_delta_traj=wrist_delta_traj,
        rew_max=np.asarray(rew_max_log[: len(timeline)]),
        rew_mean=np.asarray(rew_mean_log[: len(timeline)]),
        terminated_early=np.array(len(timeline) < N),
        n_ref=np.array(N),
    )
    print(f"  Saved: {out_path}")
    if mp_env is not None:
        mp_env.close()


def _build_weights(args) -> RewardWeights:
    """Allow CLI override of individual reward weights, plus a `--preset`.
    All weights are for L2-norm errors (m, rad). See reward.py."""
    presets = {
        # SPIDER default (pos_rew_scale=1.0, rot_rew_scale=0.3, contact=0).
        "default": RewardWeights(),
        # Same as default but with contact reward ON, scaled comparably.
        "contact": RewardWeights(
            cube_pos=1.0, cube_quat=0.3, finger_qpos=1.0,
            contact=1.0, wrist_delta=0.5,
        ),
        # Strong cube tracking + contact, SPIDER's humanoid-style override.
        "object_heavy": RewardWeights(
            cube_pos=3.0, cube_quat=1.0, finger_qpos=1.0,
            contact=1.0, wrist_delta=0.5,
        ),
    }
    w = presets.get(args.preset)
    if w is None:
        raise SystemExit(f"unknown --preset: {args.preset}; choices: {list(presets)}")
    for field in ("cube_pos", "cube_quat", "finger_qpos", "contact", "wrist_delta"):
        v = getattr(args, f"w_{field}")
        if v is not None:
            setattr(w, field, v)
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8,
                        help="multiprocess pool size; 1 = sequential")
    parser.add_argument("--preset", default="default",
                        choices=["default", "contact", "object_heavy"])
    parser.add_argument("--w-cube-pos", type=float, default=None)
    parser.add_argument("--w-cube-quat", type=float, default=None)
    parser.add_argument("--w-finger-qpos", type=float, default=None)
    parser.add_argument("--w-contact", type=float, default=None)
    parser.add_argument("--w-wrist-delta", type=float, default=None)
    parser.add_argument("--finger-noise", type=float, default=None,
                        help="MPPI finger ctrl noise std (rad). 0 = open-loop IK replay.")
    parser.add_argument("--wrist-noise", type=float, default=None,
                        help="MPPI wrist axis-angle noise std (rad).")
    parser.add_argument("--contact-always", action="store_true",
                        help="Ignore the per-frame contact mask: pull every "
                             "valid-target finger toward its contact target "
                             "throughout the sequence.")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Virtual-stick warmup substeps before MPPI: cube "
                             "is pinned and fingers settle into a stable grip. "
                             "0 disables. ~50 substeps ≈ 100 ms.")
    parser.add_argument("--warmup-closure", type=float, default=0.0,
                        help="Per-joint closure bias (rad) applied to finger ctrl "
                             "during warmup. Forces position-controlled fingers "
                             "to squeeze into the cube. ~0.2 = mild, 0.5 = firm.")
    parser.add_argument("--loop-closure", type=float, default=0.0,
                        help="Per-joint closure bias (rad) applied to finger ctrl "
                             "during the whole MPC loop (both world step and "
                             "rollouts). Maintains grip force throughout, not "
                             "just during warmup.")
    parser.add_argument("--drop-threshold", type=float, default=0.10,
                        help="If cube position error exceeds this (m), terminate "
                             "MPPI early. The cube has fallen out and no MPPI "
                             "tuning will recover it.")
    args = parser.parse_args()

    weights = _build_weights(args)
    if args.contact_always:
        weights.contact_always_on = True
    cfg = MPPIConfig(
        num_samples=args.num_samples,
        horizon_steps=args.horizon,
        max_iters=args.iters,
        weights=weights,
    )
    if args.finger_noise is not None:
        cfg.finger_noise_std = args.finger_noise
    if args.wrist_noise is not None:
        cfg.wrist_noise_std = args.wrist_noise

    print(f"  weights: cube_pos={weights.cube_pos}  cube_quat={weights.cube_quat}"
          f"  finger_qpos={weights.finger_qpos}  contact={weights.contact}"
          f"  wrist_delta={weights.wrist_delta}")

    for d in args.session_dirs:
        print(f"\n{d.name}  [workers={args.workers}]")
        run_session(d, force=args.force, cfg=cfg, seed=args.seed,
                    num_workers=args.workers, warmup_steps=args.warmup_steps,
                    warmup_closure=args.warmup_closure,
                    loop_closure=args.loop_closure,
                    drop_threshold_m=args.drop_threshold)


if __name__ == "__main__":
    main()
