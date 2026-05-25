# Dynamic retargeting — status & forward plan

> Living handoff doc for the SPIDER-style retargeting pipeline on the `spider-rewrite`
> branch. Supersedes the "Phase 4" status notes in `spider_rewrite_plan.md`.
> Last updated: 2026-05-20 (recapture + cleanup session).

## TL;DR

The offline retargeting pipeline (human cube manipulation → Tesollo DG-5F trajectory)
**works end-to-end and completes the full motion.** Stage 3 dynamic MPC produces real
in-hand cube rotation that tracks the reference for the *entire* sequence with **no
drop** — `frac_term=0.00` throughout on both validated motions:

| motion | N (mjwp) | pos err mean / max | rot err mean / max | drop? |
|---|---|---|---|---|
| **cube_roll2**  | 512  | 24.8 / 51.0 mm | 3.1° / 47.8° | no |
| **cube_pitch2** | 2048 | 11.3 / 27.2 mm | 1.3° /  9.3° | no |

cube_pitch2 is the motion that **dropped at the hardest regrip in the prior session** —
it now tracks cleanly to t=200. This falsifies the earlier "Tesollo kinematic limit"
hypothesis: **it was the capture quality, not the robot.**

**What actually fixed it (2026-05-20, recapture session):** the three motions were
**re-recorded**, holding a **stable grasp for ~2 s at the start AND end**, moving
**slower**, and **not sliding the fingers** on the cube. Because the motion is now
robot-feasible, the whole stack of drop-era band-aids (cube_pos deadband + quadratic
wall, closure-bias warmup/loop terms, stale-rotation early-stop) became unnecessary and
was **deleted**. The validated weights are now the defaults.

**Calibration offsets are now off by default-of-practice:** retargeting was run with
`--no-offsets` (Stage 2) — the equality IK is robust without them and the hand cramps
differently on the robot anyway, so the per-pose offsets weren't helping dynamically.

(Prior load-bearing fix, still true: Stage 3 uses a **real freejoint cube**, not the
virtual-stick 3-slide+3-hinge Euler joints whose `armature`/`frictionloss` had tripled
the cube's rotational inertia and crippled rotation.)

## Pipeline shape (current)

```
Stage 0   Calibration (offsets)            optimize_wrist_offset / optimize_finger_offsets
Stage 1   Processing                       → processed_sequence_with_proxy.npz
Stage 2   SPIDER equality IK               retarget_sequence_spider.py → retargeted_sequence.npz
Stage 2.5 Contact detection (cube faces)   spider_contact.py → contact_sequence.npz
Stage 3   Sampling MPC (freejoint cube)    optimize_sequence_dynamic.py → retargeted_sequence_dynamic.npz
Replay    replay_dynamic.py                --stage2 mode for kinematic; default for dynamic
```

### Stage 2 — equality IK (`retarget_sequence_spider.py`, `build_ik_scene.py`)
- Equality springs: fingertip CONNECT + wrist/cube WELD. Collisions **off** (kinematic).
- Fingertip targets are **cube-relative** (from the recording), so they track the cube
  regardless of wrist placement. Wrist offset only moves the palm.
- **Interior-only surface snap** (`tesollo_target_mapping._clamp_target_outside_cube`):
  targets that land inside the cube are pushed to `cube_half + 5mm` on the nearest face
  (5mm = fingertip geom half-size, so the tip geom just-touches without penetrating).
  Exterior/hovering targets are **left alone** — a mild hover is gentler than forcing
  contact where physics won't allow it, and lets spider_contact's SD-mask correctly
  mark those frames not-in-contact.
- `--cube-shift X Y Z` (wrist frame, metres): shifts the cube target before IK so fingers
  wrap a cube held away from the palm. Saved as `cube_shift` key. Stage 3 reads the
  already-shifted reference; **no Stage-3 shift flag**.
- `--no-offsets`: zero both wrist + fingertip calibration offsets. **Now the working
  default-of-practice** — the equality IK + interior snap are robust without them, and
  the hand cramps differently on the robot than the human anyway, so the per-pose offsets
  did not help dynamically. Both validated runs used `--no-offsets`.

### Stage 3 — sampling MPC (`dynamic_retargeting/`, `optimize_sequence_dynamic.py`)
- **Real freejoint cube** (no virtual stick, no cube actuators). 23-dim action:
  `[20 finger ctrls, 3 wrist axis-angle delta]`. Wrist position pinned to ref;
  orientation = ref ⊗ exp(delta).
- Backends: `cpu` (sequential MuJoCo) / `mjwp` (mujoco_warp GPU, CUDA-graph capture).
- Kinematic-pin warmup (`--warmup-steps`, default 50): cube qpos overwritten + qvel
  zeroed each substep so fingers settle into a grip before MPC; cube released after.
  Converts the Stage-2 kinematic "touch" pose into a force-closed physics grip so the
  free cube isn't dropping on tick 0. **Kept** — both validated runs used it.
- Reward terms (all weighted, terminal-scaled): `cube_pos` (plain world-frame L2),
  `cube_quat`, `finger_qpos`, `contact` (cube-frame relative), `wrist_delta`,
  `qvel` (**finger-qvel deviation from ref only** — wrist mocap-driven, cube must be free
  to rotate). Validated default weights: cube_pos **0.3**, cube_quat **1.0**,
  finger_qpos **0.2**, contact **1.5**, wrist_delta 0.5, qvel 0.005 — orientation-led
  tracking with a strong contact term and a light qpos pull.
- MPPI: per-component noise (finger/wrist), per-horizon-step log anneal
  (`first_step_noise 0.3 → last_step 1.0`, SPIDER convention: tight near-term, exploratory
  far-term), per-iter `beta_traj=0.7`, top-10% softmax (`temp=0.3`), zero-noise sample 0,
  resample-on-drop inside the rollout.

## Reward / MPC knobs (Stage 3 CLI)

| Flag | Default | Effect |
|---|---|---|
| `--backend mjwp` | cpu | GPU; required for N>128 |
| `--num-samples` | 256 | per MPPI iter (512 used for roll, 2048 for pitch on GPU) |
| `--iters` | 8 | MPPI iters per tick |
| `--substeps` | 25 | mj_step per ref tick (sim_dt 0.002, ref_dt 0.05) |
| `--horizon` | 16 | lookahead in ref ticks (was 8; 16 is the validated default) |
| `--w-{cube-pos,cube-quat,finger-qpos,contact,wrist-delta,qvel}` | 0.3/1.0/0.2/1.5/0.5/0.005 | reward weights (validated working config) |
| `--terminal-scale` | 3 | multiplier on H-1 step reward |
| `--time-scale` | 1 | integer reference upsample; K=2 halves Δref/tick, 2× task time |
| `--max-ticks` | 0 | cap MPC ticks for fast diagnostics |
| `--warmup-steps` | 50 | kinematic-pin grip warmup substeps |
| `--finger-noise/--wrist-noise` | 0.15/0.04 | MPPI per-component noise std |
| `--drop-threshold` | 0.10 | cube pos-err (m) for resample-on-drop |

The validated command for either motion (vary `--num-samples`):
```
.venv/bin/python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session> \
    --force --backend mjwp --num-samples 2048 --iters 8 --substeps 25 --horizon 16 \
    --w-cube-pos 0.3 --w-cube-quat 1.0 --w-finger-qpos 0.2 --w-contact 1.5 --w-qvel 0.005
```
(the `--w-*` values now match the defaults, so a bare run reproduces them.)

Stage 3 saves per-tick diagnostics in the npz: `pos_err_history`, `rot_err_history`,
`iter_improvement_history`, `term_names`, `term_means_history`, plus `cube_ref_used` /
`qpos_ref_used` (the reference MPC optimized against — used by replay so the ghost frame
count matches even with `--time-scale`).

### How to read the per-tick console line
```
t=  N/total  rew=R (improv +I)  pos=Pmm  rot=D°  frac_term=F
             terms: cube_pos=..  cube_quat=..  finger_qpos=..  contact=..  ...
```
- `rew`, `terms`: pure-penalty, **closer to 0 = better**. Each term is `-weight × error`.
- `pos`/`rot`: world-frame cube error vs the current-tick reference. **Smaller = better.**
- `improv = rew_max[last iter] − rew_max[first iter]`: **positive = MPPI improved the plan
  this tick**. Per-tick metric; do NOT compare `rew` across ticks (different ref windows).
- `frac_term`: fraction of rollout samples that dropped the cube; 1.0 = resample-on-drop
  has no good samples (degenerate). On the validated runs this stays **0.00 throughout**.

## Current results (recapture session, 2026-05-20)

Both motions complete the full sequence with `frac_term=0.00` at every tick — the cube is
never dropped, and the per-term penalties stay flat (no end-of-run spike).

| motion | N | iters | horizon | pos err mean / max | rot err mean / max |
|---|---|---|---|---|---|
| cube_roll2  | 512  | 8 | 16 | 24.8 / 51.0 mm | 3.1° / 47.8° |
| cube_pitch2 | 2048 | 8 | 16 | 11.3 / 27.2 mm | 1.3° /  9.3° |

cube_pitch2 (which dropped at the hardest regrip last session) now tracks to t=200 with
mean rot err **1.3°**. The dominant remaining penalty is `contact` (~-2.0 mean on pitch),
i.e. the relative-contact target is the term with the most headroom — not a failure mode,
just where tracking is loosest.

## Diagnosis (resolved)

The prior "robot kinematic limit" hypothesis was **wrong** — or at least not binding. The
same Tesollo hand reproduces the recaptured cube_pitch with no drop. The real culprit was
**capture quality**: the original recordings had fast multi-finger regrips and finger
sliding that produced a reference no in-hand controller could track. Re-recording with a
stable 2 s grasp at start/end, slower motion, and no finger sliding made the reference
robot-feasible, and the existing pipeline tracks it.

Practical takeaway for future captures: **the recording technique matters more than any
reward knob.** Hold static grasps at the ends, move slowly, don't slide the fingers.

## Forward plan

### 1. Re-run + eval cube_yaw2 (immediate)
roll + pitch are validated. Run `optimize_sequence_dynamic` on `recordings/cube_yaw2`
with the same command and confirm `frac_term=0.00` + low rot err. If it also passes, all
three motions are done.

### 2. Task reward — now OPTIONAL (not needed for current results)
Only pursue if a future, harder motion can't be tracked by trajectory following. Idea
unchanged: keep `finger_qpos` + `contact`, reward the rotation *motion* (angular velocity
/ axis) instead of exact per-frame cube pose (DexCap / HORA style). Lower priority now
that trajectory tracking works on feasible captures.

### 3. Other levers if needed (lower priority)
- Domain randomization in the rollout (SPIDER does this; we don't) for sim-to-real
  robustness — relevant once we move toward hardware, not for sim tracking.
- Friction tuning (cube torsional already bumped 0.02 → 0.3; `solref`/`solimp` unexplored).

## Session changelog (2026-05-20, recapture + cleanup)
- **Recaptured all three motions** (stable 2 s grasp at start/end, slower, no finger
  sliding). cube_roll2 + cube_pitch2 now track the full sequence with no drop.
- Ran retargeting with **`--no-offsets`** — confirmed offsets aren't needed dynamically.
- Promoted the validated weights to **defaults**: cube_pos 1.0→0.3, cube_quat 0.3→1.0,
  finger_qpos 0.3→0.2, contact 1.0→1.5; `--horizon` default 8→16.
- **Deleted the drop-era band-aids** (no longer needed; suno branch is the rollback):
  - `cube_pos` deadband + quadratic wall (reward.py fields + `--cube-pos-deadband`,
    `--cube-pos-wall-start`, `--cube-pos-wall-scale`); cube_pos is plain world-frame L2 again.
  - closure bias: `set_closure_bias`/`_flexion_closure_direction`/`ctrl_bias` in both envs,
    `--warmup-closure`, `--loop-closure`, `closure_bias` param on `warmup_grasp`.
  - stale-rotation early-stop (`--stale-rotation-window`, `--stale-rotation-deg`).
  - `object_heavy` preset + `--preset` flag (defaults now cover the validated config).
  - **Kept**: `warmup_grasp` (touch→grip, both runs used it), resample-on-drop +
    `--drop-threshold`, `--time-scale`, `--max-ticks`.

### Prior changelog (2026-05-20, earlier — virtual-stick → freejoint session)
- **Reverted virtual stick → real freejoint** in Stage 3 (cpu_env, mjwp_env, build_ik_scene,
  mppi, optimize_sequence_dynamic). Action 29-dim → 23-dim. This unlocked rotation.
- Reward fixes: flipped horizon noise schedule (0.3→1.0), qvel as finger-deviation only,
  `terminal_scale` 10→3, `softmax_temp` 0.3.
- Added per-term reward breakdown + pos/rot/improvement logging + npz keys.
- Added `--time-scale` (reference upsample) and `--max-ticks` (fast diagnostic cap).
- Moved `--cube-shift` from Stage 3 to Stage 2 (IK re-solves fingers for the shifted cube).
- Surface snap reworked to **interior-only** with 5mm margin in tesollo_target_mapping.
- Added `--no-offsets` flag to Stage 2.
- replay_dynamic: `--stage2` mode + `--cube-shift` green-ghost preview + `cube_ref_used`.
- Cube torsional friction 0.02 → 0.3 in `assets/output_mocap_physics.xml`.
```
