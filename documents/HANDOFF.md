# Project handoff — current status

> Quick cross-session/cross-PC status. Last updated: 2026-05-20.
> Branch: `spider-rewrite`. For full pipeline detail read
> `documents/dynamic_retargeting_status.md` first; this doc is the *current*
> state on top of it (verification + SPIDER-match work).

## ⚠️ Before using this on another PC

**All of this session's work is uncommitted in the working tree.** To pick it up
elsewhere you must first commit + push the `spider-rewrite` branch here, then pull
it there. Env: uv-managed `.venv` (use `.venv/bin/python`, not `pip`); `mjwp`
backend needs an NVIDIA GPU (warp-lang + mujoco-warp); cpu backend works anywhere.

## Where we are

The offline retargeting pipeline (human cube manipulation → Tesollo DG-5F) works
end-to-end through Stage 3 (sampling MPC). cube_roll2 + cube_pitch2 tracked the
full motion under the *original* config. This session added **Stage 4
(verification)** and discovered the trajectories are **fragile** — they don't
survive a physics change — then rebuilt the Stage-3 setup to match SPIDER's
(robust-by-design) configuration to fix it. That rebuild is **implemented but not
yet validated** — the deciding experiment (a ctrl_dt sweep + cross-backend verify)
still needs to run on GPU.

## What this session changed (working tree, uncommitted)

### New tooling
- **`scripts/evaluation/verify_open_loop.py`** — Stage 4. Replays the saved
  feedforward controls (`ctrl_traj`+`wrist_delta_traj`) with no replanning;
  reports tracking-vs-reference + divergence-from-plan. `--backend cpu` on an
  mjwp-planned run = sim-to-sim transfer test. `--save-tag` writes a replayable npz.
- **`scripts/evaluation/compare_dynamic.py`** — reward-agnostic metric table
  across runs (success, rotation coverage, rot/pos err, contact, effort, t/s).
- **`--tag NAME`** on `optimize_sequence_dynamic` → `retargeted_sequence_dynamic__<tag>.npz`
  (no clobber) + full run config stamped into the npz (`run_config`).
- **`--tag` / `--no-dense`** on `replay_dynamic`.

### Verification findings (the "why we changed everything")
- Open-loop replay reproduces the plan exactly on the **same** backend, so the
  saved trajectory IS the feedforward result — but:
  - **cube_pitch2**: cube held but **rotation collapses** cross-backend (mjwp→cpu).
  - **cube_roll2**: **drops even same-backend** (knife-edge, zero margin).
- Root cause is **not** lack of domain randomization. It's **stiff contacts +
  fine 0.05 s control** making the dynamics chaotic/exact-physics-dependent.
- Config gaps found & fixed in the cpu verifier: integrator/iterations are no-ops
  for this model; **`impratio` is real** — mujoco-warp ignores it (uses 1.0), the
  XML asked for 10. Residual cpu↔mjwp gap after matching = float32-vs-float64 +
  warp's separate solver impl (not configurable).

### SPIDER comparison (`src/spider`)
- SPIDER **ships DR off** (`num_dr: 1` in all example configs). Robustness comes
  from **physics + control setup**, not DR:
  - **soft contacts** (`o_solimp` dmin=0, width=0.03, "softer contact for sim2real")
  - **condim 4** on object↔thumb/index (torsional friction; ours was 3 → cube
    spun freely on a fingertip)
  - **coarse control** `ctrl_dt = 0.4 s` (vs our 0.05 s) — 4 control knots over a
    1.6 s horizon, **interpolated** (smooth), not zero-order-hold
  - `sim_dt 0.01`, `num_samples 1024–2048`, `iters 16–32`, `temperature 0.1`.

### SPIDER-match implemented (now the defaults)
- **Physics** (`cpu_env.apply_spider_physics`, both backends; toggle
  `--no-spider-physics`): `sim_dt 0.01`, soft `geom_solimp [0,0.95,0.03,0.5,2]`,
  `condim 4` on cube, friction `[1,1,0.1]`, `impratio 1`, implicit-fast.
- **Control-rate decoupling** (`_resample_reference_to_dt`): one MPC tick = one
  control step = `ctrl_dt = substeps·sim_dt = 0.4 s`; the reference is resampled
  onto that grid. Coarse control = the key robustness lever.
- **Sampling** (`MPPIConfig` defaults): `num_samples 1024`, `iters 16`,
  `horizon 4`, `num_knots 4`, `temperature 0.1`, noise `0.5→1.0`, finger 0.15 /
  wrist 0.03, beta from `final_noise 0.1`.
- **Control interpolation + dense recording**: `step_world` interpolates the
  applied control `plan[0]→plan[1]` (SLERP for the wrist) across substeps — smooth
  spline, not a 0.4 s staircase — and records a dense per-substep trajectory
  (`dense_qpos/dense_cube/dense_timeline`, ~50 fps) so replay is smooth.
- **Reward**: added optional `cube_angvel` term (default weight 0) for task/motion
  tracking; raise `--w-cube-angvel` and zero `--w-cube-quat` to switch from pose to
  motion tracking (cube_pos still anchors position).

### Current Stage-3 defaults (a bare run = full SPIDER match)
`sim_dt 0.01`, `substeps 40` (ctrl_dt 0.4 s), `horizon 4`, `num_knots 4`,
`num_samples 1024`, `iters 16`, `temperature 0.1`; weights cube_pos 0.3 /
cube_quat 1.0 / finger_qpos 0.2 / contact 1.5 / wrist_delta 0.5 / qvel 0.005 /
cube_angvel 0; SPIDER physics ON.
**Note:** earlier `__angvel`/`canonical` results were under the OLD hard-contact /
fine-control regime — re-plan to compare.

## What still needs to be done

1. **Run the ctrl_dt sweep on GPU** (commands below) and the cross-backend verify.
   Open question: does coarse control + soft contacts shrink the cpu↔mjwp gap and
   make roll same-backend-reproducible? Find the dexterity-vs-robustness sweet spot
   (0.1 / 0.2 / 0.4 s).
2. **Re-plan pitch + roll under SPIDER defaults** (`--tag spider`) and verify — the
   deciding test of whether the rebuild fixed fragility *without* DR.
3. **Evaluate cube_yaw2** — never run through Stage 3.
4. **If still fragile → domain randomization** (last resort; SPIDER's is worst-case
   min-reward over `num_dr` contact-param sets — `src/spider/spider/optimizers/sampling.py:276`).
   Our model has `npair=0`, so DR would randomize geom_friction/solref, not pair_margin.
5. **Minor consistency gap**: control interpolation is applied in `step_world`
   (execution) but the rollout still zero-order-holds within a control step — make
   the rollout consistent if the mismatch matters.
6. **Commit/clean up**: lots of experiment artifacts in `recordings/*/` (`*_ol_*`,
   `*_aligned`, `angvel_ol_*`); prune. Nothing committed yet.

## Key commands

```bash
# Stage 3 plan (SPIDER-matched defaults), tagged
.venv/bin/python -m scripts.retargeting.optimize_sequence_dynamic recordings/<sess> \
    --backend mjwp --tag spider

# ctrl_dt sweep (holds horizon-time ~1.6s; vary control granularity)
for cfg in "40 4 ctrl0p4" "20 8 ctrl0p2" "10 16 ctrl0p1"; do set -- $cfg
  .venv/bin/python -m scripts.retargeting.optimize_sequence_dynamic recordings/<sess> \
    --backend mjwp --num-samples 2048 --substeps $1 --horizon $2 --tag $3 --force
done

# open-loop / sim-to-sim verify (cpu replay of a gpu plan)
.venv/bin/python -m scripts.evaluation.verify_open_loop recordings/<sess> \
    --tag <tag> --backend cpu --save-tag <tag>_ol

# compare runs / replay (dense smooth playback)
.venv/bin/python -m scripts.evaluation.compare_dynamic recordings/<sess> --tags <t1> <t2> ...
.venv/bin/python -m scripts.retargeting.replay_dynamic recordings/<sess> --tag <tag>

# A/B against the old regime
.venv/bin/python -m scripts.retargeting.optimize_sequence_dynamic recordings/<sess> \
    --backend mjwp --no-spider-physics --sim-dt 0.002 --substeps 25 --horizon 16 --tag old
```

## Key files
- `scripts/retargeting/optimize_sequence_dynamic.py` — Stage 3 driver (loop, resample, save)
- `scripts/retargeting/dynamic_retargeting/{mppi,reward,cpu_env,mjwp_env}.py` — MPC core
- `scripts/retargeting/replay_dynamic.py` — viewer
- `scripts/evaluation/{verify_open_loop,compare_dynamic}.py` — Stage 4 + analysis
- `assets/output_mocap_physics.xml` — base scene (single source of truth; SPIDER
  physics is applied programmatically in the env, not here)
