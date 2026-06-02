# Pipeline — overview, current status, what's left

> Living handoff for the SPIDER-style retargeting pipeline on the `spider-rewrite`
> branch. The thesis outline lives in `thesis/paper_outline.md`.

## Overview

End-to-end pipeline: **Manus Quantum Gloves + dual-camera ArUco → calibration →
20 Hz processed sequence → SPIDER-style equality IK → sampling MPC → MuJoCo
replay**, targeting the Tesollo DG-5F. Reproduces the SPIDER §2.1-2.3 method on a
new embodiment via a self-built capture and calibration stack.

## Pipeline stages

| Stage | Purpose | Driver |
|---|---|---|
| 0 | Calibration: wrist-to-board + fingertip-to-cube offsets | `scripts/calibration/{estimate_wrist_to_board,optimize_hand_object_alignment}.py` |
| 1 | Processing: resample 20 Hz, outlier reject, low-pass, wrist transforms | `scripts/processing/build_sequence_dataset.py` → `processed_sequence_with_proxy.npz` |
| 2 | Equality-constraint IK (SPIDER §2.1) | `scripts/retargeting/retarget_sequence_spider.py` → `retargeted_sequence.npz` |
| 2.5 | Per-frame cube-face contact detection | `scripts/retargeting/spider_contact.py` → `contact_sequence.npz` |
| 3 | Sampling MPC over a real freejoint cube (SPIDER §2.2-2.3) | `scripts/retargeting/optimize_sequence_dynamic.py` → `retargeted_sequence_dynamic[__<tag>].npz` |
| Replay | Per-tick MuJoCo replay | `scripts/retargeting/replay_dynamic.py` |
| Compare | Cross-run metric table | `scripts/evaluation/compare_dynamic.py` |

## Stage 3 architecture

### Control rate and physics

`sim_dt = 0.005`, `substeps = 10`, ⇒ `ctrl_dt = 0.05 s` (20 Hz control,
200 Hz physics). `horizon = 16` steps (0.8 s lookahead). `num_knots = horizon`
(no knot interpolation). Implicit-fast integrator.

Cube physics: `condim 3` (sliding-only; torsional and rolling DOFs left
free — pinch-pitch rotation depends on them), friction `[1.0, 0.1, 0.1]`
(torsional/rolling values ignored at condim 3), stiff `solimp
[0.9, 0.99, 0.001]`, `impratio` inherited from XML (mjwp ignores it).

Action: 23-dim `[20 finger ctrls, 3 wrist axis-angle delta]`. Wrist position
pinned to reference; orientation = ref ⊗ exp(delta).

### Execution semantics — zero-order hold

`env.step_world(plan[0])` applies `plan[0]` and the wrist mocap target
**unchanged across all n_substeps**, matching what `rollout_with_resample`
simulates when scoring plans. The previous interpolated execution was a bug:
the planner forecast was systematically wrong because the rollout assumed ZOH
but execution interpolated `plan[0] → plan[1]`. Fixed in
`{cpu,mjwp}_env.step_world`. `next_action` / `next_ref_*` arguments are
accepted for API compatibility but discarded.

### MPPI loop (per tick)

Per iter (`max_iters = 12` default, early-stoppable):
1. Sample N noisy control sequences around current plan.
2. Sample 0 = zero-noise prior. Last `exploit_ratio × N` samples have
   noise scaled by `exploit_noise_scale = 0.01` (near-greedy fallback).
3. Per-horizon-step noise anneal `first_step_noise_scale = 0.5 →
   last_step_noise_scale = 1.0` (tight near-term, exploratory far).
4. Per-iter global noise anneal via `beta = final_noise_scale^(1/max_iters)
   ≈ 0.825` per iter; `global_scale = beta^iter` (1.0 → 0.1).
5. Snapshot world state into N batch worlds; rollout H steps in parallel.
6. Resample-on-drop: per horizon step, any sample whose cube pos > 10 cm or
   rot > 30° from reference gets its state + action history replaced with a
   surviving sample's. (`drop_threshold_m = 0.10`,
   `drop_threshold_rad = π/6 = 30°`.)
7. Score: per-step linear penalty for `cube_pos`, `cube_quat`,
   `finger_qpos`, `contact`, `wrist_delta`, `qvel`, `cube_angvel`.
   Terminal step (h=H-1) is **quadratic** in error for `cube_pos`,
   `cube_quat`, `finger_qpos` (SPIDER-style sharp endpoint), multiplied
   by `terminal_scale = 10`. Other terms linear × terminal_scale.
8. Update plan = top-10% softmax-weighted average of samples;
   `softmax_temp = 0.3`, `top_frac = 0.1`.
9. **Early stop**: break if `rews.max() − rews[0] < improvement_threshold
   = 0.01` (SPIDER's convergence check — best exploration this iter didn't
   beat the prior plan).

After the iter loop: **always** rollout the chosen plan as a single extra
rollout (broadcast plan to N samples, take sample 0) to log its actual
reward + per-term values into `info["plan_rew"]`, `info["plan_terms"]`. With
`--diagnostics`, the same eval runs after **every** iter so the plan-rew
trajectory is queryable from the npz.

After `optimize_step`: apply `plan[0]` via `step_world` (ZOH), measure realized
pos / rot error, advance world.

### Top-level run-loop early-stop

If `frac_terminated == 1.0` (every rollout sample dropped its cube), the
top-level loop in `optimize_sequence_dynamic.run_session` breaks and saves
the partial trajectory. Compute is not wasted on dead ticks.

### Reward defaults (in `RewardWeights`)

`cube_pos 1.0`, `cube_quat 0.6`, `finger_qpos 0.05`, `contact 2.0`,
`wrist_delta 0.3`, `qvel 0.01`, `cube_angvel 0.25`, `terminal_scale 10.0`.
(Working base runs override `finger_qpos 0.02`, `wrist_delta 0.15`,
`qvel 0.005` via CLI; new defaults reflect later tuning, not the base sweep
state — pin via CLI when reproducing.)

## Saved npz fields (per Stage 3 run)

Per-tick:
- `timeline`, `retargeted_qpos`, `cube_traj`, `cube_ref_used`, `qpos_ref_used`
- `pos_err_history`, `rot_err_history` (realized, against cube_ref_used)
- `iter_improvement_history` (rew_max[-1] − rew_max[0])
- `term_means_history` (chosen plan's terminal per-term values)
- `plan_rew_history` (chosen plan's total reward, scalar per tick)
- `ctrl_traj`, `wrist_delta_traj`

Per-tick × per-iter (NaN-padded to `max_iters` where early-stopped):
- `rew_max`, `rew_mean` (per-iter best/mean sample reward)
- `plan_rew_per_iter` (chosen plan's reward at each iter; with
  `--diagnostics` is dense, otherwise has the last-iter value at one column)

Metadata:
- `run_config` (JSON string): every knob, weights, MPPI params, timings.
- `term_names`.

Diagnostic-only console output: realized pos/rot, plan rew, best-sample rew,
MPPI improv, frac_term, plan terms (forecast). Per-tick output every
`print_every` ticks (10 by default, 1 with `--diagnostics`).

## CLI flags worth remembering

| Flag | Default | Notes |
|---|---|---|
| `--num-samples` | 1024 | |
| `--horizon` | 16 | |
| `--num-knots` | = horizon | |
| `--iters` | 12 | |
| `--substeps` | 10 | |
| `--sim-dt` | 0.005 | |
| `--finger-noise` | 0.10 | |
| `--wrist-noise` | 0.03 | |
| `--drop-threshold-deg` | 30 (set in MPPIConfig) | rot drop |
| `--drop-threshold` | 0.10 m | pos drop |
| `--w-*` | reward.py defaults | individual weight overrides |
| `--terminal-scale` | 10 | |
| `--seed` | 0 | |
| `--tag NAME` | none | output goes to `retargeted_sequence_dynamic__<NAME>.npz` |
| `--force` | off | overwrite existing tag |
| `--diagnostics` | off | per-tick print + per-iter plan eval (~2× compute) |
| `--no-spider-physics` | physics on | A/B against original hard-contact regime |

## Status

**Working end-to-end** on all three motions (cube_roll2, cube_pitch2,
cube_yaw2) under the base config (N=1024, H=16, iters=12). 5-seed base runs
show 5/5 success on pitch and yaw, 4/5 on roll (seed 3 consistently drops
under current config — characterized as the real failure case rather than
retried). MPPI sweep on cube_pitch2 complete: 60/60 success across
samples / horizon / iters axes.

Per-seed determinism is high — given a fixed numpy seed, the trajectory
repeats. GPU FP noise plays a smaller role than initially assumed; seed-to-
seed variance dominates run-to-run.

## What still needs to be done

Ordered by dependency.

1. **Implement physics-parameter CLI flags.**
   - `--cube-edge EDGE_M`: resize cube geom; recompute inertia for uniform
     density. Add to Stage 2 (`retarget_sequence_spider.py`), Stage 2.5
     (`spider_contact.py`), and Stage 3 (`optimize_sequence_dynamic.py`).
   - `--cube-mass MASS_KG`: Stage 3 only (no kinematic effect on Stages 2 /
     2.5).
   - Stage 2/2.5 reruns per edge variation produce edge-specific
     `retargeted_sequence.npz` / `contact_sequence.npz`. Stored under tagged
     filenames (e.g. `retargeted_sequence__edge0p04.npz`).
2. **Implement late-start CLI flag.**
   - `--start-time SECONDS` on Stage 3: trim reference (qpos_ref,
     cube_ref, contact_target) by `int(start_time / ref_dt)` frames, set
     initial world state to the trimmed-frame reference. Warmup_grasp
     kept on (the experiment is "does a mid-motion start need a stable
     grasp anchor at t=0 of MPC?", not "can the planner acquire grip
     from free-fall").
3. **Rerun cube_roll2 seed 3 with alternative seeds** (5, 6) to obtain a
   non-dropping 5-seed set. The original seed 3 failure remains documented
   as the real failure rate.
4. **Physics-parameter sweep, cube_pitch2 (§4.3)**.
   - Stage 2 + 2.5 reruns for edges 0.04, 0.05, 0.07, 0.08 m. (~3 min total)
   - Stage 3: edges 0.04, 0.05, 0.07, 0.08 m × 5 seeds = 20 runs at mass
     baseline (×1).
   - Stage 3: masses ×0.5, ×2, ×4 × 5 seeds = 15 runs at edge baseline
     (0.06 m).
   - Total: 35 Stage 3 runs.
5. **Late-start experiment (all three motions)**.
   - Trim depths: 3.0 s, 4.5 s, 6.0 s.
   - All three motions × 3 trims × 5 seeds = 45 runs.
6. **Notebook for thesis figures.** `compare_dynamic.py` prints tables;
   need a notebook that loads the same npz files and produces:
   per-tick error plots, Pareto curves for §4.2, heatmap for §4.3,
   bar charts for late-start.
7. **Final cleanup before submission.**
   - Prune experiment artifacts in `recordings/*/`: `*_ol_*`, `*_aligned`,
     `angvel_ol_*`, stale tagged npz.
   - Audit reward.py: `cube_angvel` term defaults to nonzero now (0.25);
     keep documented as a motion-tracking lever even if the main runs use
     pose-only tracking.
   - Commit the working tree.

## Metrics (all derivable from `cube_traj` / `cube_ref_used` / `run_config`)

- `success` — bool, no drop (max pos err ≤ drop_threshold AND completed full
  ref length).
- `rot_cover` — cumulative-rotation sim / ref ratio. <1 = undershot, >1 =
  oscillated.
- `pos_err` / `rot_err` mean / max / final.
- `fin_geo` — geodesic orientation error at the final tick.
- `t/s` — ticks per second (compute cost; the §4.2 x-axis).
- `wall_s` — total wall-clock time.

## Key files

| Path | Role |
|---|---|
| `scripts/retargeting/optimize_sequence_dynamic.py` | Stage 3 driver (loop, save, top-level early-stop) |
| `scripts/retargeting/dynamic_retargeting/mppi.py` | MPPI iter loop, early stopping, plan eval helpers |
| `scripts/retargeting/dynamic_retargeting/{cpu,mjwp}_env.py` | World + batch envs; `step_world` (ZOH) and `rollout_with_resample` |
| `scripts/retargeting/dynamic_retargeting/reward.py` | Per-term reward, terminal quadratic for tracking terms |
| `scripts/retargeting/retarget_sequence_spider.py` | Stage 2 IK |
| `scripts/retargeting/spider_contact.py` | Stage 2.5 contact detection |
| `scripts/retargeting/replay_dynamic.py` | Per-tick MuJoCo replay |
| `scripts/evaluation/compare_dynamic.py` | Cross-run metric table |
| `assets/output_mocap_physics.xml` | Base scene; physics mutations applied programmatically in env |

## Common commands

```bash
# Stage 3 with current defaults
.venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic recordings/<sess> \
    --backend mjwp --tag <name>

# Diagnostic run — tick-by-tick + per-iter plan eval (~2× compute)
.venv/bin/python -u -m scripts.retargeting.optimize_sequence_dynamic recordings/<sess> \
    --backend mjwp --diagnostics --tag diag

# Compare runs
.venv/bin/python -m scripts.evaluation.compare_dynamic recordings/<sess> \
    --tags <t1> <t2> ...

# Replay (per-tick)
.venv/bin/python -m scripts.retargeting.replay_dynamic recordings/<sess> --tag <tag>
```

Use `python -u` when piping to `tee` — Python switches to block-buffered
stdout otherwise and you won't see progress.

## Environment

uv-managed `.venv`; use `.venv/bin/python`, not `pip`. The `mjwp` backend
requires an NVIDIA GPU (warp-lang + mujoco-warp); `cpu` backend works
anywhere but is much slower at high N.
