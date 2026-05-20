# SPIDER-style rewrite plan

This document is the **starting brief for a fresh session** that rewrites the offline retargeting pipeline along SPIDER's architectural lines while keeping the cube-specific advantages we've built (face segmentation, manual constraints).

> **Status (2026-05-20):** Phases 0-3 + 5 done. Stage 3 reworked — virtual stick
> removed, real freejoint cube restored (this unlocked in-hand rotation). cube_pitch now
> tracks ~70° rotation for ~170 ticks then drops at the hardest regrip; likely a Tesollo
> kinematic limit. **See `dynamic_retargeting_status.md` for the current architecture,
> knobs, results, and forward plan (recapture easier motions; possibly a task reward).**
> The "Phase 4 actual status" section below is superseded by that doc.

## Why we're rewriting

We accumulated a stack of bespoke workarounds — Pass-1 cube smoother, Pass-2 feasibility IK with snap-to-surface, whole-finger CollisionAvoidanceLimit, A→B→A merge, smooth blend weights — because our gradient-IK (mink) can't gracefully relax when targets are infeasible. SPIDER avoids the entire problem by using **MuJoCo physics with equality-constraint springs** for IK; the same simulator handles both IK and MPC. Their MPC also has critical structural pieces ours lacks: a **virtual stick** on the object that prevents drop-at-start, **resample-on-drop**, **per-component noise scaling**, and a **2048-sample budget**. The result is robust enough to handle 6 datasets × multiple robots out of the box.

Our cube_roll dropping at t=2 isn't a tuning problem — it's the absence of these pieces.

## What we keep, what we replace, what we delete

### Keep (don't touch)
- **Calibration**: `optimize_wrist_offset.py`, `optimize_finger_offsets.py` — these are independent of the retargeter and proven.
- **Stage 1 processing**: `scripts/processing/*` produces `processed_sequence_with_proxy.npz`. Untouched.
- **Runtime mink IK**: `scripts/apps/run_tesollo_teleop_*.py` use mink for live teleop. Different latency/safety requirements; keep mink there.
- **Contact-schedule visualization**: `scripts/visualization/plot_contact_schedule.py`. Just update the input file name when Stage 2 changes.
- **Face segmentation + hysteresis + A→B→A merge + manual constraints**: all in `spider_contact.py`. These extend SPIDER's contact detection to handle cube faces, which their averaging approach can't.
- **`manual_contacts.json` format and loader**: `scripts/retargeting/manual_contacts.py`. Already integrated.
- **Cube collision geom naming** (per-finger phalanx names in `output_mocap.xml` / `output_mocap_physics.xml`).

### Replace
- **Stage 2a (cube_relative IK)** → SPIDER-style equality-constraint IK.
- **Stage 3 (MPPI optimizer)** → SPIDER-style sampling MPC: 2048 samples, 16 iters, resample-on-drop, per-component noise, zero-noise first sample, virtual stick.

### Delete
- **Stage 2c (feasibility IK)**: `scripts/retargeting/retarget_sequence_feasible.py`, the `feasible_retargeting/` submodule (`cube_smoother.py`), `replay_feasible.py`. The new equality IK handles infeasibility natively, snap-to-surface is unnecessary, the Pass-1 cube smoother is replaced by a weak object equality constraint.
- **Whole-finger collision avoidance pairs** in `tesollo_constants.py` (`FEASIBILITY_NONTIP_PAIRS`, `FEASIBILITY_TIP_PAIRS`).

### Target pipeline architecture (after rewrite)

```
Stage 0: Calibration (one-time)
    optimize_wrist_offset.py    →  DEFAULT_WRIST_OFFSET constants
    optimize_finger_offsets.py  →  OFFLINE_TARGET_OFFSETS dict

Stage 1: Recording → processed_sequence_with_proxy.npz   (existing)

Stage 2: SPIDER-style kinematic retargeting   [NEW — replaces 2a + 2c]
    retarget_sequence_spider.py
      ├── Build IK scene XML on the fly (mocap target bodies +
      │       equality constraints for fingertips, palm, cube)
      ├── Disable collisions (default)
      ├── Random initial guess for frame 0 (8 candidates → best)
      ├── Per frame: set mocap targets, run mj_step N times
      └── Output: retargeted_sequence.npz
                  {timeline, retargeted_qpos (N,27),
                   cube_pose_used (N,7)}

Stage 2.5: Contact detection on Stage 2 output   (existing, adapted)
    spider_contact.py
      ├── geom-collision mask + SD-near-surface fallback
      ├── face classification with hysteresis
      ├── face-based segmentation + A→B→A merge
      ├── manual_contacts.json overrides (face-bound / no-contact)
      └── Output: contact_sequence.npz

Stage 3: Sampling-based MPC (dynamic retargeting)   [REWRITE]
    optimize_sequence_dynamic.py
      ├── Scene: same physics XML + object xyz/rpy actuators ("virtual stick")
      ├── 256-2048 samples (CPU MuJoCo, scale to budget; MJX optional)
      ├── 16 iters per MPC tick with horizon-axis noise annealing
      ├── Per-component noise (pos/rot/joint scales decoupled)
      ├── First sample = zero-noise (preserves current plan)
      ├── Resample-on-drop: replace dropped-cube samples mid-rollout
      ├── Virtual stick: object-actuator action decays from frame 0
      └── Output: retargeted_sequence_dynamic.npz
```

## Step-by-step implementation guide

Each phase is independent and can be merged separately. Run the smoke-tests at the end of each before moving on.

### Phase 0: Snapshot & branch (30 min)

1. `git status`; commit the current working tree (smooth snap + manual constraints + visualization).
2. `git checkout -b spider-rewrite`.
3. Skim `src/spider/spider/preprocess/ik.py` (lines 1–600) and `src/spider/spider/optimizers/sampling.py` once end-to-end. The IK file is the load-bearing reference.

**Done-when**: clean branch off `suno`; you can locate `add_mocap_bodies`, `make_optimize_once_fn`, and `_compute_weights_impl` in SPIDER source from memory.

### Phase 1: Scene XML with mocap targets and object actuators (1 day)

Create a script `scripts/retargeting/build_ik_scene.py` (or hand-author the XML once and check in) that produces `assets/output_ik_physics.xml`:

1. Start from `assets/output_mocap.xml`.
2. Add a mocap body + sphere site for each of: 5 fingertips (`thumb_tip`, …), wrist palm site, cube center.
3. Add `<equality>` constraints:
   - **Fingertips** → `CONNECT` (position-only), `solref=[0.01, 1.0]`, `solimp=[0.0, 0.95, 0.01, 0.5, 2.0]`.
   - **Wrist palm** → `WELD` (pose), `solref=[0.02, 1.0]`, `solimp=[0.0, 0.95, 10.0, 0.5, 2.0]`, `torquescale=10`.
   - **Cube** → `WELD`, `solref=[0.002, 1.0]`, `solimp=[0.9, 0.95, 0.001, 0.5, 2.0]`, `torquescale=10`.
   - SPIDER source for exact param values: `src/spider/spider/preprocess/ik.py:342-360`.
4. Also produce `assets/output_dyn_physics.xml` — same scene **without** equality constraints but **with** cube position/rotation actuators (3 slide + 3 hinge replacing the freejoint, kp/kd small) for Stage 3. SPIDER pattern in `src/spider/spider/preprocess/generate_xml.py:26-130`.

**Done-when**: `mujoco.MjModel.from_xml_path(...)` loads both scenes cleanly. `mj_forward` with mocap bodies set to arbitrary positions produces sensible equality solver behavior (palm and fingers move toward the mocaps).

### Phase 2: SPIDER-style IK (2 days)

Create `scripts/retargeting/retarget_sequence_spider.py` replacing both `retarget_sequence_cube_relative.py` and `retarget_sequence_feasible.py`.

Algorithm (per session):

```
load processed_sequence_with_proxy.npz → fingertips_cube[N,5,3], wrist_world[N,4,4], cube_pose_world[N,7], ergonomics[N,20]
build mocap-aware IK scene (Phase 1)
mj_data = MjData; disable collisions, disable actuation

# Random initial guess for frame 0 (SPIDER's max_num_initial_guess=8)
best_q = None
best_residual = inf
for k in range(8):
    qpos = random_in_joint_limits()
    set_mocap_targets(frame=0)  # cube + fingertips + palm
    for _ in range(30):
        mj_step()
    if mocap_residual() < best_residual:
        best_q = qpos.copy()
mj_data.qpos = best_q

# Frame-by-frame: warm-start from previous, step physics
for t in 0..N-1:
    set_mocap_targets(frame=t)
    for _ in range(K_steps):   # e.g. K=30, sim_dt=0.01 → 300 ms of relaxation per ref tick
        mj_step()
    qpos_out[t] = mj_data.qpos[:27]
    cube_pose_out[t] = mj_data.qpos[27:34]
```

Mocap targets come from:
- Fingertips: `R_world_cube @ (raw_fingertip_cube * scale + offset)` (same calibration as before, applied through `map_cube_frame_target` semantics).
- Palm/wrist: `T_world_wrist @ wrist_offset` (existing `_apply_wrist_offset`).
- Cube: recorded cube pose world.

Output `retargeted_sequence.npz` with the same key names that `spider_contact.py` already reads (`retargeted_qpos`, `cube_pose_used`, `timeline`).

**Done-when**:
- Runs end-to-end on both cube_pitch and cube_roll without errors.
- The output `retargeted_sequence.npz` can be visualized in MuJoCo viewer at 30 fps and roughly matches the recorded motion.
- Per-frame fingertip → IK-target distance < ~10 mm (similar to current Stage 2a).
- The contact-schedule plot (Stage 2.5 → visualization) looks no worse than current.

### Phase 3: Adapt contact detection (half a day)

Update `spider_contact.py`:
1. `detect_contact_session` already supports `from_feasible=True/False`. Rename to `from_stage_2c/from_stage_2a` → `from_spider_ik` reading `retargeted_sequence.npz`. Drop the Stage 2c fallback.
2. Update the file name in `optimize_sequence_dynamic.py`'s loader to read `retargeted_sequence.npz`.
3. Visualization (`plot_contact_schedule.py`) — same change.

**Done-when**: `python -m scripts.retargeting.spider_contact recordings/cube_pitch` runs on Stage 2 output; `contact_schedule.png` regenerates with the new data; manual constraints still applied.

### Phase 4: MPC rewrite (3-5 days)

This is the biggest single piece. Stay in `scripts/retargeting/optimize_sequence_dynamic.py` for the entry point but rewrite the optimizer internals.

Key changes (each independently testable):

1. **Per-component noise** (`scripts/retargeting/dynamic_retargeting/mppi.py`):
   - Decompose `cfg.noise_scale` into `pos_noise_scale` (3), `rot_noise_scale` (3), `joint_noise_scale` (20), `object_pos_noise_scale` (3), `object_rot_noise_scale` (3).
   - Apply horizon-axis logspace annealing as SPIDER does (`first_ctrl_noise_scale` → `last_ctrl_noise_scale`).
   - Set **first sample to zero noise** (`noise_scale[0] *= 0`) so the prior plan is always evaluated.

2. **Sample budget**: bump `num_samples` default to 256. On CPU MuJoCo with `workers=8`, expect 256 × H=8 × substeps=25 ≈ 51200 sim steps per MPC tick. Profile expected runtime; can scale to 1024 if budget permits or move to MJX.

3. **Resample-on-drop** (`scripts/retargeting/dynamic_retargeting/mppi.py`, inside the rollout loop):
   - After each horizon step, check `terminate[i] = cube_pos_err[i] > drop_threshold`.
   - Replace `bad_indices` with random sampling from `good_indices`: swap full sim state + ctrl history up to the current step + cum_rew.
   - SPIDER reference: `src/spider/spider/optimizers/sampling.py:141-178`.

4. **Virtual stick on object** (new):
   - Scene loaded from `output_dyn_physics.xml` (Phase 1) has 6 object actuators (3 slide + 3 hinge replacing the cube freejoint).
   - Extend the action vector to 23 + 6 = 29 dims: [20 finger ctrls, 3 wrist delta, 3 obj pos, 3 obj rot].
   - The 6 object-actuator gains start high (kp=10) and **decay over the MPC sequence** (geometric decay e.g. ×0.5 every 50 frames, or annealed by `i / max_num_iterations` per MPPI iteration as SPIDER does).
   - First MPC tick the cube is heavily supported; by frame ~30 the actuators are effectively off and fingers must hold.

5. **Cross-entropy weighting** (`mppi.py`):
   - Keep top-10% softmax (already present), just verify it matches SPIDER's `_compute_weights_impl`.

6. **Reward**: keep the existing SPIDER-style reward in `reward.py`. Adjust the contact-target source to handle our face-segmented per-frame targets (already in place).

**Done-when**:
- cube_pitch runs 200/200 frames with cube_pos_err < 30 mm mean, cube_rot_err < 30° mean.
- cube_roll runs 200/200 frames (no drop) with the same quality bands.
- The same hyperparameters work for both sessions (no per-session tuning).

### Phase 5: Cleanup (half a day)

Delete:
- `scripts/retargeting/retarget_sequence_cube_relative.py` (replaced by Stage 2)
- `scripts/retargeting/retarget_sequence_feasible.py`
- `scripts/retargeting/feasible_retargeting/` (entire directory)
- `scripts/retargeting/replay_feasible.py`
- `tesollo_constants.FEASIBILITY_NONTIP_PAIRS`, `FEASIBILITY_TIP_PAIRS`, `FEASIBILITY_COLLISION_PAIRS`
- The `_apply_manual_constraints` plumbing in retarget_sequence_feasible (moved to spider_contact only)

Rename:
- `replay_dynamic.py` stays
- Stage 2 replay: add `replay_spider.py` reading `retargeted_sequence.npz`

Update consumers:
- All teleop apps still import `_apply_wrist_offset`, `OFFLINE_TARGET_OFFSETS`, etc. from `retarget_sequence_cube_relative.py`. Move these constants to a small `tesollo_calibration.py` module that both runtime (mink teleop) and the new Stage 2 import.

### Phase 6: Verification (2 days)

Run the full pipeline on both sessions, compare against the original `recordings/<session>/processed_sequence_with_proxy.npz` for ground truth motion:

```
python -m scripts.retargeting.retarget_sequence_spider recordings/cube_pitch recordings/cube_roll
python -m scripts.retargeting.spider_contact recordings/cube_pitch recordings/cube_roll --stats
python -m scripts.visualization.plot_contact_schedule recordings/cube_pitch recordings/cube_roll
python -m scripts.retargeting.optimize_sequence_dynamic recordings/cube_pitch recordings/cube_roll --preset contact
python -m scripts.retargeting.replay_dynamic recordings/cube_pitch
```

Sanity-check metrics (per session):
- IK fingertip-to-target err: mean < 10 mm, max < 30 mm
- Dynamic cube pos err: mean < 30 mm, max < 80 mm
- Dynamic cube rot err: mean < 30°, max < 90°
- Dynamic contact: ≥2-finger contact > 80%
- No early termination

## Risk register

| Risk | Mitigation |
|---|---|
| Equality-IK is slow per-frame (mj_step × 30) | Profile early. If too slow, batch IK over the whole sequence using MJX, or reduce K_steps and verify quality holds. |
| Random initial guess fails to converge | If `best_residual` after 8 tries is high (e.g. > 50 mm), increase `max_num_initial_guess` to 16 or seed from the ergonomics pose. |
| Sample budget 256 on CPU is too slow | Move just the rollout to MJX. The optimizer loop stays in numpy/python; only the simulation batch is on GPU. SPIDER reference: `src/spider/spider/simulators/mjwp.py`. |
| Virtual stick breaks contact dynamics (cube can't actually be grasped) | Decay the object actuators over the rollout. By the last horizon step they should be essentially off. Mid-rollout `terminate = cube_pos_err > drop_threshold` still applies. |
| Resample-on-drop discards too much exploration | SPIDER's heuristic: only resample when *some* but *not all* samples terminate. If all dropped, accept and replan from a corrected state. |
| Removing mink IK breaks teleop apps | The runtime teleop scripts have their own mink usage; only the *offline* retargeting moves to equality-IK. Verify teleop apps still launch (`python -m scripts.apps.run_tesollo_teleop_world`). |

## Open questions for the rewrite session

1. **Inheritance vs new scene XML**: do we copy `output_mocap.xml` once with the IK additions, or generate the IK-ready scene programmatically from the base XML? SPIDER does the latter (`generate_xml.py`); the former is simpler for one robot.
2. **MJX or CPU MuJoCo for the rollout batch?** MJX gives 10–100× sample budget for ~1 day of porting work. CPU is more familiar and avoids a JAX dependency on the rollout.
3. **Per-finger force scaling**: SPIDER scales the equality constraint torque per-finger (ring/pinky weaker than thumb/index). Worth replicating? Probably yes — robot kinematic limits differ per finger.

## Out-of-scope (won't be touched)

- The Manus glove driver / processing pipeline (`src/manus/`)
- Calibration scripts
- The teleop apps in `scripts/apps/`
- Recording tools

---

## Reference points in SPIDER source

When in doubt, look at these files in `src/spider/spider/`:

- **Equality IK setup**: `preprocess/ik.py:30-380`. Watch `add_mocap_bodies` and the `site_joint_matches` dict.
- **Random initial guess**: `preprocess/ik.py:493-560`. The `max_num_initial_guess` loop.
- **Scene XML generation with object actuators**: `preprocess/generate_xml.py:22-130`. `_add_object_xyzrpy_actuators`.
- **Contact detection**: `preprocess/detect_contact.py:355-440`. The averaging pattern we extend with face segmentation.
- **MPC core**: `optimizers/sampling.py` end to end. `make_rollout_fn`, `make_optimize_once_fn`, `make_optimize_fn`.
- **Resample-on-drop**: `optimizers/sampling.py:141-178`.
- **Per-component noise**: `config.py:330-374` (`get_noise_scale`).
- **Cross-entropy weights**: `optimizers/sampling.py:204-237`.
- **CUDA Graph capture for mjwp.step**: `simulators/mjwp.py:52-68` (`_compile_step` + `wp.ScopedCapture`). 11× per-step speedup at small N.
- **Reward terms**: `simulators/mjwp.py:275-351` (`get_reward`, `get_terminal_reward`). Note `qvel_rew` and `terminal_rew_scale`.

---

## Phase 4 actual status (2026-05-17)

Open question 1 (XML inheritance): resolved → programmatic via `MjSpec`.
Open question 2 (MJX or CPU): resolved → went **mujoco_warp** (`mujoco-warp` PyPI), CUDA-graph capture for ~11× per-step speedup. CPU backend kept as fallback (`--backend cpu`).
Open question 3 (per-finger torque): partially done — fingertip `solimp` width scaled ×3 for ring/little.

### What landed in the MPC code

- Action layout 29-dim: `[20 finger ctrls, 3 wrist axis-angle delta, 3 obj_pos, 3 obj_rot]`.
- Per-component noise (decoupled finger/wrist/obj scales, defaults `0.15 / 0.04 / 0.0 / 0.0` rad — obj noise *off* by default because it's wasted with stick=0 and counterproductive with stick=1).
- Per-iter `beta_traj=0.7` and per-horizon-step log anneal `1.0 → 0.3`.
- Zero-noise sample 0 (prior plan always evaluated).
- Top-10% softmax weighting (`_topk_softmax`).
- Resample-on-drop inside the horizon loop (`rollout_with_resample`), threshold via `--drop-threshold`.
- **Iter-based virtual stick** (`iter_stick_schedule`, default `warm_then_off`): iter 0 = full stick, iters 1+ = stick off. **World step always uses stick=0** — stick is an optimizer warm-start, never real-physics aid.
- Wrist-frame `--cube-pos-deadband` (2 cm bubble of zero penalty around the ref).
- `--cube-shift X Y Z` (wrist-frame translation of cube ref) — for when the recorded cube starts too close to the palm.
- `--w-qvel` velocity penalty (default 0.05) — encourages stable equilibrium.
- `--terminal-scale` (default 10) — multiplies the H-1 step reward to fix myopic drift.
- `--stale-rotation-window` early termination.

### What's not working yet

cube_pitch / cube_roll at N=2048 × 8 iters track well for ~50 frames then drift over the sequence; cube drops around t=140-170. Diagnosis (last session, see memory):
- Myopic MPC drift: per-tick plan looks fine locally but accumulates error. Terminal reward fix landed but **not yet run**.
- IK fingertip pose is "touching the cube surface" not "wrapping for force closure"; even modest squeeze pushes the cube. `--warmup-closure 0` is essential (otherwise the warmup squeeze shoves the cube 49 mm off-ref).

### Next-session run

```
.venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic recordings/cube_roll \
  --force --backend mjwp --num-samples 2048 --iters 8 --substeps 25 \
  --cube-shift 0.02 0 -0.01 --cube-pos-deadband 0.02 \
  --w-finger-qpos 0.3 --w-cube-quat 1.0 --w-contact 1.0 \
  --w-qvel 0.05 --terminal-scale 10 \
  --warmup-closure 0.0 --loop-closure 0.2 \
  --iter-stick-schedule warm_then_off
```

If still drifting: bump `--terminal-scale 30`, or `--w-qvel 0.2`, or accept some cube push with `--loop-closure 0.4`.
For cube_pitch specifically (in-hand rotation): also try `--horizon 16` (doubles runtime).
