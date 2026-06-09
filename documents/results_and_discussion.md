# Thesis notes — Results and Discussion

Draft prose for the §4 Results and §5 Discussion sections. Numbers reference
`documents/sweep_analysis.md`. Figures referenced live in
`documents/figures/`. Adapt freely for the final paper.

---

## §4 Results

### §4.1 Base evaluation (three motions, five seeds)

We evaluated the dynamic retargeting pipeline on three in-hand cube
rotations recorded with the Manus glove + ArUco tracking stack: a
pitching, a rolling, and a yawing motion. Each motion was processed
through Stages 0–2.5 once, and Stage 3 was run with five seeds per
motion under the baseline SPIDER-style configuration (N = 1024 samples,
horizon H = 16, M = 12 MPPI iterations). All 15 runs completed the full
reference trajectory without dropping the cube. Per-motion summary
statistics are given in Table 1 (sweep_analysis.md §"Base runs").

Per-tick rotation error (Figure: `base_per_tick.png`, top row)
stays below 10 ° throughout all three motions. Mean trajectory error is
2.2° for pitch and roll, 3.3° for yaw — a moderate degradation on the
motion with the largest perpendicular content (see §5).

To diagnose the *path quality* of the planned trajectories, we
decomposed each per-tick rotation increment into a component along the
reference motion's dominant axis and the orthogonal complement
(Figure: `wobble_per_tick.png`). The along-axis coverage $C_\parallel$
is close to one for pitch (1.07) and roll (1.19), confirming the
planner reproduces the captured rotation. The perpendicular coverage
$C_\perp$ is 1.8 for pitch, 2.0 for yaw, and **3.2** for roll, meaning
the planner adds substantial incidental rotation around axes the human
did not use. This wobble is examined in the Discussion.

### §4.2 MPPI compute-vs-tracking sweep

We swept the three principal MPPI hyperparameters (sample count N,
horizon H, refinement iterations M) one at a time around the base
configuration on cube_pitch2, with five seeds per point. The Pareto
view (Figure: `mppi_pareto.png`, rows share y-axis for fair
inter-axis comparison) shows:

- **N has the strongest effect on mean tracking error** ($\bar e_\text{rot}$
  spans 1.9–3.4 ° across N ∈ {128, …, 2048}).
- **H has the weakest effect**: a 3× horizon range produces only a
  0.3 ° spread in mean error.
- **iters has the strongest effect on endpoint pose error**
  ($e_\text{geo}^\text{fin}$ spans 2.1–7 ° across M ∈ {2, …, 16}). In
  particular, M = 8 produces the *best* endpoint accuracy of any
  configuration tested, beating both the M = 12 baseline (4.8 °) and
  M = 16 (7.0 °).

These observations identify $N = 512$, $H = 20$, $M = 8$ as a
Pareto-informed candidate configuration that should reduce wall-clock
time without sacrificing tracking. We validate this candidate in §4.4.

### §4.3 Physics-parameter sweeps

#### Cube edge length

The captured motion's mocap and IK were specified for a 70 mm cube.
We swept the simulated cube edge to {50, 60, 80, 90} mm with five seeds
each, re-running Stages 2 and 2.5 to keep contact targets cube-frame
correct (Table in sweep_analysis.md §"Physics sweep — cube edge").

Smaller cubes track reliably: 5/5 success at 60 mm, 4/5 at 50 mm.
**Larger cubes fail abruptly**: 1/5 success at 80 mm, 0/5 at 90 mm. The
failure is not gradual — between 70 mm and 80 mm the planner loses the
ability to keep grip. Endpoint geodesic error climbs from 4.8 °
(base, 70 mm) to 15.3 ° at 90 mm.

#### Cube mass

Mass was swept at {20, 80, 160} g with the geometry held at the
captured 70 mm. All 15 runs succeeded. Tracking quality improves
monotonically with mass: mean rotation error drops from 2.13 ° (20 g)
to **1.89 ° (160 g)** — better than the 40 g baseline (2.20 °) —
and endpoint geodesic error drops from 4.14 ° to **1.41 °**, a
3× improvement at the heaviest mass tested. The per-seed variance also
collapses: rot_mean σ falls from 0.22 (20 g) to 0.07 (160 g).

A torque-budget sanity check on the 160 g runs (commanded vs. achieved
joint deflection) found median deflection ≈ 0.06 rad and p99 ≈ 0.33 rad,
well below saturation given the actuator force ranges in the model. The
160 g performance is therefore in the unconstrained regime, not at the
edge of physical feasibility.

### §4.4 Optimal-config validation

The candidate Pareto-informed configuration from §4.2 ($N = 512$,
$H = 20$, $M = 8$) was run with five seeds across all three motions,
holding reward weights and physics identical to the base. Results are
shown in Table "Optimal config vs base" of sweep_analysis.md and in
Figure: `opt_vs_base.png`.

Headline numbers:

- **Wall-clock time falls by 27–33 %** across all three motions
  (444 → 303 s on pitch; 645 → 469 s on roll; 719 → 482 s on yaw).
- **Mean rotation error degrades modestly** (+9 % pitch, +19 % roll,
  +10 % yaw).
- **Endpoint pose accuracy improves substantially for pitch (−63 %) and
  roll (−49 %)** but regresses for yaw (+21 %).
- **Per-seed standard deviation collapses across the board**: rot_mean
  σ drops 55-73 % for all three motions, wall-time σ drops 33-74 %.
  The optimal configuration is markedly more consistent than the base.
- Reliability is preserved: 5/5 success on both base and opt for all
  three motions.

---

## §5 Discussion

### §5.1 Why iters = 8 beats the baseline iters = 12

The MPPI iteration loop refines a single plan by repeatedly sampling
N noisy candidates around the current plan, ranking by reward, and
forming a new plan as the softmax-weighted average of the top samples.
With per-iteration noise annealing the per-step noise shrinks over the
loop, so late iterations sample tightly around the previous plan.

In the diagnostic mode we observed an *averaging-collapse* failure
mode at late iterations: the per-iteration best sample's reward
improves monotonically across iterations, but the softmax-averaged
plan—rolled out separately—sometimes degrades sharply because the top
samples represent two or more *qualitatively different* feasible
solutions, and their weighted average lies in a no-man's-land that
neither sample explored. This is a known pathology of sample-mean
trajectory optimization.

The Pareto data confirms the effect: iters = 16 produces strictly worse
endpoint pose error (7.0°) than iters = 8 (2.1°) on the cube_pitch2
sweep, despite drawing twice the samples. Eight iterations are enough
to converge MPPI on this problem; further iterations introduce more
averaging-collapse opportunities than they fix.

### §5.2 Why heavier cubes track better

The monotonic improvement of tracking with cube mass is consistent
with four overlapping mechanisms:

1. **Inertia damps the cube's response to MPPI sample noise.** Per-tick
   force perturbations produce angular accelerations inversely
   proportional to inertia; a 4× heavier cube responds 4× less to
   the same sample-noise-induced force, suppressing wobble while
   leaving the *commanded* trajectory intact.
2. **Stronger gravity pinning aids force closure.** Cube weight loads
   each contact's normal force directly. At 160 g the friction-cone
   constraints are satisfied more comfortably during fast regrips,
   making slip less likely.
3. **Better numerical conditioning under stiff contacts.** Position
   drift from per-step integration error scales inversely with mass at
   constant contact stiffness.
4. **Tighter feasible-plan basin.** With the cube less reactive, the
   set of MPPI plans that produce acceptable trajectories is smaller
   and more uniquely defined, so different random seeds converge to
   similar plans (the seed-to-seed σ on rot_mean falls from 0.22
   at 20 g to 0.07 at 160 g).

The captured demonstration did not specify a cube mass; the 40 g XML
default is a placeholder. The sweep suggests that a heavier operating
mass would simplify the planning problem without violating any
physical constraint in the simulator. For a hardware deployment, the
implication is that the Tesollo would benefit from manipulating
moderately heavy objects rather than very light ones.

### §5.3 The edge-length feasibility cliff

The captured fingertip positions (Stage 2 IK output) are expressed
relative to a 70 mm cube. When the simulated cube is shrunk, the
captured grip pattern still fits — the fingers can wrap a smaller cube
with the geometry they were taught — and we observe near-perfect
success down to 50 mm. When the cube is enlarged the captured grip
no longer reaches around it, and the planner cannot recover: 80 mm
already fails 4/5 and 90 mm fails 5/5, with endpoint pose error
exploding from 4.8° (base) to 15.3°.

This is a boundary of the **kinematic reference**, not of the
**dynamic planner**. The MPPI itself is doing the same job at 90 mm
as at 50 mm; what changes is the feasibility of the IK pose targets
the contact reward is pulling toward. A more general pipeline would
re-derive contact targets for each cube size from the captured
*relative-to-cube-surface* finger placements rather than the absolute
fingertip positions. We did not pursue this generalization within
the scope of this thesis.

### §5.4 Off-axis wobble has a structural origin

The base perpendicular coverage $C_\perp$ values of 1.8–3.2 mean the
planned cube trajectories add 80–220 % more incidental rotation than
the captured demonstrations did. The effect is most pronounced on the
cube_roll motion, where the captured perpendicular content is small
(the human roll is a clean motion) but the planner adds a comparable
amount of rocking.

A plausible mechanism: MPPI's per-finger Gaussian noise is isotropic
on the *control space* (joint position targets), but the *cube* has
free rotational degrees of freedom and no axial symmetry constraint
on the applied torques. Small asymmetries in the per-tick finger
noise integrate into a slowly drifting torque around the cube's
unconstrained rotational DOFs, accumulating as rotation around axes
the planner is not penalizing as strongly.

The endpoint pose remains accurate ($e_\text{geo}^\text{fin}$ of
4–5° on base) — the cube ends up where it should — but the path
taken by the cube is wigglier than the human's. For trajectory-replay
purposes this is acceptable; for hardware execution it would warrant
attention because the wobble would manifest as extra mechanical work
on the finger joints.

### §5.5 Why yaw behaves differently

The yaw motion is the only one of the three where:
- the base perpendicular coverage is close to the along-axis coverage
  (1.98 vs 1.63), suggesting the captured yaw motion already has
  significant off-axis content;
- the optimal config *regresses* on endpoint pose error;
- and per-seed variability on the base config is the highest of the
  three motions.

We hypothesize this stems from the orientation of the dominant
rotation axis: for yaw the axis is approximately vertical, so neither
gravity nor the captured grip geometry strongly constrains rotation
around it. Pitch and roll rotations are at least partially
gravity-loaded — the cube wants to fall back into a stable grip —
whereas yaw rotation is in a near-symmetric energy landscape. The
planner has more equally-good feasible options, and small differences
in MPPI sampling lead to qualitatively different plans across seeds.

For practical purposes this means yaw is a harder motion than pitch
or roll for this hardware, not because the rotation is mechanically
harder but because the *space of feasible plans is wider* and less
well-conditioned.

### §5.6 Limitations and scope

- **Simulation only.** No claim is made about transfer to physical
  Tesollo hardware. GPU non-determinism inside mujoco-warp means that
  even at fixed numpy RNG seed, bit-identical reproducibility is not
  guaranteed; for hardware transfer the planner would also face
  contact-model mismatch, actuator dynamics not present in the
  MuJoCo model, and force/torque limits at the joint level that the
  simulator's actuator force ranges only approximate.
- **One cube geometry, one fabric of motions.** The edge sweep
  exposes the feasibility limit of the captured grip; we did not
  evaluate other shapes (non-cubic) or motions that require
  multi-finger regrips. The "edge cliff" at 80 mm is specific to this
  capture.
- **Five seeds per configuration.** The std bars in all tables are
  estimated from this small sample. Reported values should be
  interpreted as point estimates with σ as a rough scale, not as
  inputs to formal statistical tests.
- **No domain randomization.** All runs were performed under a single
  physics parameterization. A robustness sweep over friction,
  contact-stiffness, and mocap noise was not within the scope of this
  thesis.
- **Capture-protocol dependency.** Earlier in the project we observed
  that mid-motion-start trials (skipping the captured stable grasp
  at the beginning of the recording) failed reliably. The current
  results depend on the recording protocol of holding a static grasp
  at the start and end of each motion.
