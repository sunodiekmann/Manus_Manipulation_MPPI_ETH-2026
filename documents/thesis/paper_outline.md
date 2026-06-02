# Thesis — paper outline

> ETH semester thesis. Short report format, target 10-15 pp, single column.
> Companion to `../pipeline.md` (engineering status). This doc is the *paper*
> plan: titles, sections, contributions, experiments, metrics.

## Format

- ETH semester thesis, **single-column report**, 11 pt, ~25 mm margins.
- Preferred: `scrartcl` with `biblatex-ieee`. If the supervisor's lab (ASL /
  RSL / IRIS / Soft Robotics Lab) supplies a template, use that instead.
- **Avoid** double-column IEEE conference style — too compressed for 10-15 pp.

## Candidate titles (decide late)

- **A.** *Learning dexterous manipulation from human demonstrations*
  - Original framing. If kept, the abstract must clarify that "learning" here
    means *learning a control sequence per demonstration via sampling-based
    optimization*, not training a policy.
- **B.** *Scaling dexterous retargeting across compute and object physical parameters*
  - Tighter to what §4.2 + §4.3 actually do.

## Contributions to claim (intro)

1. **End-to-end capture-to-feasible-trajectory pipeline for the Tesollo
   DG-5F**, integrating Manus gloves + dual-camera ArUco tracking, two-stage
   calibration, and a faithful re-implementation of SPIDER's sampling MPC.
2. **Compute-vs-tracking trade-off characterization** of SPIDER-style MPPI on
   a new embodiment (§4.2 single-axis sweeps over samples, horizon, iters at
   5 seeds each).
3. **Physics-parameter scaling on one demonstration**: feasibility-vs-
   parameter map for the captured cube rotation across cube edge length and
   cube mass (§4.3, reproducing SPIDER §2.5 augmentation on Tesollo).

## Empirical findings (discussion)

**Finding A — stable grasp at sequence endpoints is required.** Without a
static ~2 s grasp at start and end, the freejoint cube exits the IK warmup
with non-zero velocity and the MPC cannot recover the initial grip. The
pipeline **is** robust to other capture noise sources (Manus drift, ArUco
jitter, sub-cm camera/calibration error — all visible in Stage 2 replay,
absorbed by Stage 3). The mid-motion late-start experiment (§4.4) tests
whether MPPI can pick up an in-progress rotation when the start frame has
fewer fingers in contact than the original full grasp.

**Finding B — execution-rollout consistency is load-bearing.** An early
version of the pipeline applied interpolated controls in execution
(`step_world` blended `plan[0] → plan[1]` over substeps) while the MPPI
rollout zero-order-held each plan step. Plans that scored well in the
rollout produced different physics under execution; the cube would drop at
ticks where the planner's forecast said it was safe. Fixing execution to
match rollout (both ZOH) removed an entire class of "the plan looked fine
then the cube dropped" failures. *Methodological lesson: the
planner-execution physics must be bit-equivalent.*

**Finding C — MPPI averaging collapse is a real, observable failure mode.**
Even with consistent physics, the per-iter diagnostic shows ticks where the
best individual sample improves monotonically across iters (good MPPI
convergence) but the softmax-weighted plan update lurches into a
significantly worse rollout than the previous iter produced. The
chosen-plan-vs-best-sample gap quantifies this directly. A principled fix
is "return best plan across iters", which the thesis recommends as future
work since it doubles per-tick rollout cost.

**Finding D — `iters` carries a hidden quality trade-off.** Mean tracking on
the §4.2 sweep is best at `iters=8`, but worst-tick excursions (`rot_max`)
are best at `iters=12-16`. More iters do tighten convergence but expose
more averaging-collapse opportunities. The choice between base configs
depends on whether mean or worst-case tracking matters more. We pick
`iters=12` as the reliability-leaning base.

## Outline

| § | Title | pp |
|---|---|---|
| 1 | Introduction | 1.5 |
| 2 | Related work | 1 |
| 3 | Methods | 4-5 |
| 4 | Results (§4.1 base, §4.2 MPPI, §4.3 physics, §4.4 late-start) | 4-5 |
| 5 | Discussion | 1.5 |
| 6 | Conclusion + future work | 0.5 |

### 1. Introduction (~1.5 pp)

Funnel: dexterous manipulation hard → human demos abundant but suffer the
embodiment gap → kinematic IK is fast but infeasible, RL is dynamic but
task-specific, physics-based retargeting (SPIDER, ManipTrans) bridges this.

Research question framing for title B: *"Can a SPIDER-style sampling MPC
scale dexterous retargeting beyond a single demonstration — across MPPI
compute budgets and across object physical parameters?"*

End with the three contributions above.

### 2. Related work (~1 pp)

Three short paragraphs, one sentence of positioning at the end.

- **Kinematic retargeting**: DexPilot, AnyTeleop, Reda et al. — fast but
  contact-unaware / dynamically infeasible.
- **RL-from-demo / residual policies**: DexCap, HORA, OpenAI Rubik's —
  dynamic but task-specific, expensive, embodiment-specific.
- **Physics-based retargeting**: SPIDER (primary anchor), ManipTrans,
  DexMachina, DIAL-MPC — the family this work is in.

Positioning: *"This work applies SPIDER's sampling MPC to a new embodiment
(Tesollo DG-5F) via a self-built capture stack, and characterizes its
compute–tracking trade-off and physics-parameter scalability."*

### 3. Methods (~4-5 pp)

#### 3.1 Capture & calibration (~1.5 pp)

- Hardware: Manus Quantum Gloves + 2× RealSense + ArUco tags (left glove,
  right glove, 6 cm cube board), fixed table board defines world frame.
- Stage 0 calibration: (a) coarse wrist-to-board from known fingertip
  positions, (b) fine hand-object alignment that puts fingertips on the
  cube surface.
- Stage 1 processing: 20 Hz resample, jump-based outlier rejection
  (6 cm / 20° thresholds), low-pass filtering, wrist-frame fingertip
  geometry.
- Figure: pipeline diagram with hardware setup inset.

#### 3.2 Kinematic retargeting — Stage 2 (~1 pp)

- Equality-constraint IK in MuJoCo: fingertip CONNECT + wrist/cube WELD
  springs. Collisions off (kinematic only).
- Cube-relative fingertip targets (tracks the cube regardless of wrist
  placement; wrist offset only moves the palm).
- Interior-only surface snap: targets inside the cube get pushed to
  `cube_half + 5 mm`; exterior targets left as-is.
- Stage 2.5: per-frame contact targets in cube-relative coordinates from
  IK output, used downstream by Stage 3's contact term.
- Brief — refer to SPIDER §2.1.

#### 3.3 Dynamic retargeting — Stage 3 (~2 pp)

- Cite SPIDER eqs (1a-c), (2), (3), Algorithm 1; describe rather than
  re-derive.
- 23-dim action `[20 finger ctrls, 3 wrist axis-angle delta]`; wrist
  position pinned to ref, orientation = ref ⊗ exp(delta).
- Reward terms: `cube_pos`, `cube_quat`, `finger_qpos` (linear per-step,
  quadratic at terminal — SPIDER-style endpoint anchor), `contact` (cube-
  frame relative), `wrist_delta`, `qvel` (finger-deviation-from-ref only),
  `cube_angvel`. Terminal scale = 10.
- MPPI: per-component noise, per-horizon-step + per-iter log-anneal,
  top-10% softmax (temp 0.3), exploit samples (last 1% of pool with noise
  × 0.01 as near-greedy fallback), resample-on-drop (10 cm pos / 30° rot),
  SPIDER-style early stopping (`improvement_threshold = 0.01`).
- **Execution semantics**: zero-order hold — `plan[0]` and the wrist
  mocap target are applied unchanged for the full `n_substeps`, matching
  the planner's rollout assumption. (See Finding B in Discussion.)
- Kinematic-pin warmup grasp (50 substeps) before MPC release.
- Three deliberate deviations from SPIDER, all documented with reasoning:
  stiff `solimp` (soft contacts let the 60 mm cube interpenetrate the
  fingertips), condim 3 (pinch-pitch rotation depends on the
  torsional-spin DOF that condim 4 would constrain), `sim_dt = 0.005`
  (stiff `solimp` needs finer integration than SPIDER's soft contacts).

### 4. Results (~4-5 pp)

#### 4.1 Tracking on captured rotations (~1.5 pp)

- Base config: `N=1024, H=16, iters=12, substeps=10, sim_dt=0.005`
  (ctrl_dt = 50 ms), reward weights per §3.3. Stated in a small parameter
  table.
- Base table over 5 seeds × 3 motions (mean ± std for: success rate,
  pos err mean / max, rot err mean / max, rot_cover, fin_geo, wall_s).
- Honest reporting: **4/5 success on cube_roll2 (seed 3 drops
  consistently), 5/5 on cube_pitch2 and cube_yaw2.** The seed-3 failure
  is reproducible and characterized as the residual failure rate of the
  current planner on the hardest motion.
- Per-tick determinism note: given a fixed numpy seed, the trajectory
  repeats — seed-to-seed variance dominates run-to-run GPU FP noise. This
  motivates the 5-seed evaluation protocol used throughout Results.
- One qualitative figure: 3 motions × 3-4 replay frames each.
- One per-tick error plot: `rot_err_history` for each motion.

#### 4.2 Compute vs tracking — MPPI sweep (~1.5 pp)

- **Motion**: cube_pitch2 (representative; planner properties transfer
  across motions, and the 5-seed base runs already confirm the chosen
  config works on roll + yaw).
- **Protocol**: single-axis sweeps around base, 5 seeds each, no
  cross-product:
  - `num_samples` ∈ {128, 256, 512, 1024 (base), 2048}
  - `horizon` ∈ {8, 12, 16 (base), 20, 24}
  - `iters` ∈ {2, 4, 8, 12 (base), 16}
- 60 runs total (4 sweep points per axis × 5 seeds × 3 axes). Base
  reused from §4.1.
- Plot: Pareto-style — x = wall-clock per tick, y = `rot_err.mean()`,
  one line per swept knob with error bars.
- Headline results to report:
  - **All 60 runs succeed** (no drops anywhere in the sweep).
  - Quality is sublinear in N: rot_mean drops 3.42° → 1.92° from N=128 to
    N=2048 (16× compute for ~45% rot improvement).
  - Horizon plateaus at H=12-20; H=24 doesn't improve over H=20.
  - `iters=8` achieves the best mean tracking (rot_mean 2.10°) but
    `iters=12-16` have lower rot_max (better worst-tick) — trade-off
    discussed (Finding D).
- Pareto knee: `N=512` (half the base wall time, 20% worse rot_mean,
  no drops) is the "thrifty" config recommended for §4.3.

#### 4.3 Physics-parameter scaling — cube edge × mass (~1.5 pp)

- **Motion**: cube_pitch2 (one motion; physics-parameter sensitivity is a
  system property, not a task property).
- **Protocol**: single-axis sweeps around base (6 cm, 40 g), 5 seeds each:
  - Edge length: {0.04, 0.05, 0.07, 0.08} m × 5 seeds = 20 runs.
  - Mass scale: {×0.5, ×2, ×4} × 5 seeds = 15 runs.
  - 35 runs total. Stage 2 + 2.5 re-run per edge variation (4 reruns;
    ~3 min total) to keep contact targets cube-frame-correct.
- Mass variations don't trigger Stage 2 reruns (kinematics unchanged).
- Plot: two bar charts — one for edge (rot_mean / success rate vs edge
  length), one for mass (rot_mean / success rate vs mass scale).
- Discussion: feasibility frontier — at what extreme does the planner
  start failing, and why (under-grip on smaller cubes, over-mass beyond
  finger torque, etc.).

#### 4.4 Late-start experiment (~1 pp)

- **Motivation**: directly tests Finding A — does MPPI with the working
  base config recover from a mid-motion start where the initial frame
  doesn't have a stable full-finger grasp?
- **Protocol**: trim depths 3.0 s, 4.5 s, 6.0 s × all three motions
  (cube_roll2, cube_pitch2, cube_yaw2) × 5 seeds = **45 runs**.
- Warmup_grasp is kept ON: the experiment isolates whether a mid-motion
  reference state is recoverable when given the same kinematic-pin
  warmup that full-trajectory runs get. (Disabling warmup tests grasp
  *acquisition*, which is a different question.)
- Headline: success rate vs trim depth, per motion. Expect monotonic
  degradation. If the deepest trim succeeds, Finding A is weaker than
  hypothesized; if it fails consistently, Finding A is corroborated.

### 5. Discussion (~1.5 pp)

- §4.2 Pareto observation: name the knee config; describe the iters trade-
  off (Finding D).
- §4.3 scalability observation: feasible region, what fails at boundaries
  and why.
- §4.4 corroboration of Finding A.
- Findings B and C: methodological lessons for the next student
  (execution-rollout consistency is load-bearing; averaging collapse is
  real; "return best plan across iters" is a principled future fix).
- Limitations: sim only, one robot, one shape class (cubes), three
  motions, no contact-force evaluation, no domain randomization, MPPI
  reliability on hardest motion is 80% (seed-3 cube_roll2 failure).

### 6. Conclusion + future work (~0.5 pp)

Hardware transfer to the physical Tesollo, task-objective reward
(`cube_angvel` already wired), larger object class, return-best-plan-
across-iters for the averaging collapse cases, robustification with DR if
needed for sim-to-real.

## Metrics

| Metric | Units | Interpretation |
|---|---|---|
| `success` (no drop) | bool | Did the cube survive? |
| `rot_cover` | unitless | Cumulative rotation sim / ref. >1 = over-rotation/wobble |
| `pos_err` mean / max | mm | Position tracking |
| `rot_err` mean / max | ° | Orientation tracking |
| `fin_geo` | ° | Geodesic orientation error at last tick |
| `wall_s`, `t/s` | s, ticks/s | Compute cost (essential for §4.2 Pareto) |

## Figures

Budget tight — keep to ~7 total.

1. Pipeline diagram (hardware → stages → output). §3.
2. Stage 2 IK formulation sketch (springs + cube target). §3.2.
3. Replay strip: 3 motions × 3-4 frames. §4.1.
4. Per-tick error plot, one line per motion. §4.1.
5. Pareto: wall-clock vs `rot_err.mean()`, one line per swept knob,
   error bars from 5 seeds. §4.2.
6. Bar chart: success rate / rot_mean vs cube edge and vs mass. §4.3.
7. Bar chart: success rate vs trim depth per motion. §4.4.

## Open decisions

- Title A vs B (decide late).
- Plotting wrapper: `--plot` flag on `compare_dynamic.py` vs separate
  notebook (recommend notebook for thesis-figure iteration).
- Whether to mention "return best plan across iters" only in future work
  or also as a small additional experiment if compute allows.
