# Dynamic retargeting — status & forward plan

> Living handoff doc for the SPIDER-style retargeting pipeline on the `spider-rewrite`
> branch. Supersedes the "Phase 4" status notes in `spider_rewrite_plan.md`.
> Last updated: 2026-05-20.

## TL;DR

The offline retargeting pipeline (human cube manipulation → Tesollo DG-5F trajectory)
works end-to-end. Stage 3 dynamic MPC now produces **real in-hand cube rotation**
(cube_pitch tracks ~70° rotation and the pitch-and-return motion for the first ~170
ticks). The remaining failure: on the hardest part of cube_pitch the cube drifts out of
position, a finger loses its grip, and the cube drops. Diagnosis points to a **robot
kinematic / motion-difficulty limit** (Tesollo reproducing a complex human regrip), not
a pipeline bug.

**The single most important fix this session: Stage 3 reverted from a virtual-stick
cube (3 slide + 3 hinge Euler joints + position actuators) back to a real freejoint.**
The Euler joints carried `armature=1e-4` + `frictionloss=1e-4` that tripled the cube's
effective rotational inertia and added stiction — that was crippling rotation. The
freejoint fixed it (rot_err at t=30 dropped ~18× : 25° → 1.4°).

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
- `--no-offsets`: zero both wrist + fingertip calibration offsets (diagnostic). Finding:
  the kinematic result still looks good without offsets — the equality IK + interior snap
  are robust, so the offsets are NOT a large error source. (Offsets still fine-tune exact
  contact point, which can matter dynamically — A/B in Stage 3 to be sure.)

### Stage 3 — sampling MPC (`dynamic_retargeting/`, `optimize_sequence_dynamic.py`)
- **Real freejoint cube** (no virtual stick, no cube actuators). 23-dim action:
  `[20 finger ctrls, 3 wrist axis-angle delta]`. Wrist position pinned to ref;
  orientation = ref ⊗ exp(delta).
- Backends: `cpu` (sequential MuJoCo) / `mjwp` (mujoco_warp GPU, CUDA-graph capture).
- Kinematic-pin warmup: cube qpos overwritten + qvel zeroed each substep so fingers
  settle into a grip before MPC; cube released after.
- Reward terms (all weighted, terminal-scaled): `cube_pos` (deadband + quadratic wall),
  `cube_quat`, `finger_qpos`, `contact` (cube-frame relative), `wrist_delta`,
  `qvel` (**finger-qvel deviation from ref only** — wrist mocap-driven, cube must be free
  to rotate).
- MPPI: per-component noise (finger/wrist), per-horizon-step log anneal
  (`first_step_noise 0.3 → last_step 1.0`, SPIDER convention: tight near-term, exploratory
  far-term), per-iter `beta_traj=0.7`, top-10% softmax (`temp=0.3`), zero-noise sample 0,
  resample-on-drop inside the rollout.

## Reward / MPC knobs (Stage 3 CLI)

| Flag | Default | Effect |
|---|---|---|
| `--backend mjwp` | cpu | GPU; required for N>128 |
| `--num-samples` | 256 | per MPPI iter (use 2048 on GPU) |
| `--iters` | 8 | MPPI iters per tick |
| `--substeps` | 25 | mj_step per ref tick (sim_dt 0.002, ref_dt 0.05) |
| `--horizon` | 8 | lookahead in ref ticks |
| `--w-{cube-pos,cube-quat,finger-qpos,contact,wrist-delta,qvel}` | 1.0/0.3/0.3/1.0/0.5/0.005 | reward weights |
| `--terminal-scale` | 3 | multiplier on H-1 step reward |
| `--cube-pos-deadband` | 0 | wrist-frame free-zone radius (m); ~0.02 for cube_pitch |
| `--cube-pos-wall-start` | 0 | distance past deadband where quadratic wall starts |
| `--cube-pos-wall-scale` | 0 | quadratic wall weight (per m²); ~1000 = hard wall |
| `--time-scale` | 1 | integer reference upsample; K=2 halves Δref/tick, 2× task time |
| `--max-ticks` | 0 | cap MPC ticks for fast diagnostics |
| `--warmup-steps/--warmup-closure/--loop-closure` | 50/0/0 | grip warmup + persistent closure bias |
| `--finger-noise/--wrist-noise` | 0.15/0.04 | MPPI per-component noise std |
| `--stale-rotation-window` | 0 | early-stop if rotation stalls |

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
  has no good samples (degenerate).

## Current results (cube_pitch, N=2048, iters=8, time-scale=2)

- First ~170 output ticks: rot tracks to 70° and back, pos in/near the 20mm deadband.
- ~t=180: cube position drifts past deadband → ring slips on the cube (cube-frame, not
  world) → 2-finger grip becomes unstable → cube drops; pos blows to 200mm+.
- `cube_pos` quadratic wall + higher contact weight delay but don't prevent the drop.
- MPC **detects** the impending drop (cube_pos term spikes ~1 tick before pos actually
  blows up) but has no recoverable plan — multiple terms spike together = unrecoverable
  physical state, not an optimizer failure.

## Diagnosis

This is most likely a **robot capability limit**, not a pipeline bug:
- The recorded human cube_pitch uses a fast multi-finger regrip.
- Tesollo DG-5F (4 DOF/finger, box fingertips, short distal segments) can't articulate
  fast enough to keep all fingers tracking their cube-frame contact points through the
  peak-rotation regrip. One finger lags on the cube surface → grip destabilises.
- Literature (SPIDER, HORA, OpenAI Rubik's, DexCap) succeeds on this class of task with
  **anthropomorphic hands** (Shadow / Allegro / LEAP) that match human articulation, and
  in several cases with **task-level rewards** rather than trajectory tracking.

## Forward plan

### 1. Recapture the three cube motions (next, by the user)
Re-record cube_pitch / cube_roll / cube_yaw with:
- A **static grasp at start and end** (clean entry/exit for MPC).
- **Slower** motion (less Δref per tick → physics has time to track).
- **Less extensive** rotation (smaller angle → less regrip demand).
- **Easier finger gait** (avoid fast multi-finger regrips Tesollo can't reproduce).

Then re-run the pipeline once (Stage 2 → 2.5 → 3) and evaluate. This directly addresses
the kinematic-limit hypothesis: an easier motion should be reproducible.

### 2. If recapture isn't enough → task reward (planned, not yet implemented)
Shift Stage 3 reward from **trajectory tracking** (match specific cube pose per frame) to
**task / motion tracking**:
- Keep `finger_qpos` and `contact` terms (still want robot to grip like the human).
- Replace per-frame absolute `cube_pos`/`cube_quat` tracking with a reward on the **motion
  itself** — e.g. cube angular velocity matches the reference rotation rate/axis, rather
  than the cube being at an exact orientation at an exact frame.
- Rationale: lets MPC find a Tesollo-feasible execution that achieves the same *motion*
  without being pinned to the human's exact cube trajectory. This is the DexCap / HORA
  approach and is well-supported in the literature for hand-mismatch retargeting.
- The deadband + wall machinery on `cube_pos` already partially decouples position; the
  task reward extends that idea to orientation/rotation.

### 3. Other levers if needed (lower priority)
- Domain randomization in the rollout (SPIDER does this; we don't) for robustness.
- Friction tuning (cube torsional already bumped 0.02 → 0.3; `solref`/`solimp` unexplored).
- Drop the calibration offsets entirely if the Stage-3 A/B (`--no-offsets`) shows they
  don't help dynamically — one less pipeline component.

## Session changelog (2026-05-20)
- **Reverted virtual stick → real freejoint** in Stage 3 (cpu_env, mjwp_env, build_ik_scene,
  mppi, optimize_sequence_dynamic). Action 29-dim → 23-dim. This unlocked rotation.
- Reward fixes earlier in session: flipped horizon noise schedule (0.3→1.0), qvel as
  finger-deviation only, `terminal_scale` 10→3, `softmax_temp` 0.3, finger_qpos default 0.3.
- Added per-term reward breakdown + pos/rot/improvement logging + npz keys.
- Added `--time-scale` (reference upsample) and `--max-ticks` (fast diagnostic cap).
- Added `cube_pos` quadratic wall (`--cube-pos-wall-start/scale`).
- Moved `--cube-shift` from Stage 3 to Stage 2 (IK re-solves fingers for the shifted cube).
- Surface snap reworked to **interior-only** with 5mm margin in tesollo_target_mapping;
  removed the bidirectional/spider_contact snap (it forced exterior fingers into contact
  and killed the natural release/regrip pattern).
- Added `--no-offsets` diagnostic flag to Stage 2.
- replay_dynamic: `--stage2` mode + `--cube-shift` green-ghost preview + `cube_ref_used`
  consumption (fixes time-scale frame mismatch).
- Cube torsional friction 0.02 → 0.3 in `assets/output_mocap_physics.xml`.
```
