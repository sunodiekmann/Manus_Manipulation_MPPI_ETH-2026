# Robohand2 — Motion Capture & Retargeting Pipeline

End-to-end pipeline for capturing human hand–object interactions with Manus Quantum Gloves and ArUco tag-based pose estimation, retargeting them to the Tesollo robotic hand, and optimising dynamically feasible trajectories in MuJoCo.

---

## Table of contents

1. [System overview](#system-overview)
2. [Installation](#installation)
3. [Hardware setup](#hardware-setup)
4. [Calibration](#calibration)
5. [Main workflows](#main-workflows)
6. [Sub-step details](#sub-step-details)
7. [Visualization](#visualization)
8. [Evaluation](#evaluation)
9. [Script reference](#script-reference)

---

## System overview

```
┌─────────────────── Pre-session (once per setup) ────────────────────┐
│  Camera calibration + Manus calibration + wrist offset calibration  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────── Stage 1: scripts/capture_motion.py ───────────┐
│  record_session  →  build_sequence_dataset               │
│                  →  compute_proxy_geometry                │
└───────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────── scripts/retarget_motion.py ──────────────────┐
│  Stage 2:   kinematic_retargeting   (SPIDER equality IK) │
│  Stage 2.5: detect_contacts         (physics replay)     │
│  Stage 3:   optimize_sequence_dynamic (MPPI MPC)         │
└───────────────────────────────────────────────────────────┘
```

**World frame**: defined by a fixed ArUco board on the table. Glove boards and the cube board are tracked relative to this frame. A calibration matrix maps each glove board to the corresponding hand's wrist joint.

**Pipeline stages:**
- **Stage 1 (capture)**: Records raw streams from the Manus gloves and pose estimator, resamples and filters to a clean 20 Hz timeline, and computes per-frame fingertip geometry.
- **Stage 2 (kinematic)**: SPIDER equality-constraint IK maps the human wrist pose and finger configuration to robot joint angles, frame by frame.
- **Stage 2.5 (contacts)**: physics replay detects which fingertips are in contact with the cube at each frame, building a contact schedule for Stage 3.
- **Stage 3 (dynamic)**: MPPI sampling-based MPC rolls out trajectories in a full physics simulation (cube as a free joint, fingers must grip under gravity) to produce dynamically feasible joint angles.

---

## Installation

```bash
git clone <this-repo>
cd robohand2
git submodule update --init --recursive
uv sync
source .venv/bin/activate
```

All commands use `python -m scripts.<module>` (absolute import paths from the repo root) and must be run with the virtual environment active.

### GPU backend (recommended)

Stage 3 (MPPI) supports an NVIDIA Warp GPU backend (`--backend mjwp`) that is significantly faster than the default CPU rollout. It requires:

- NVIDIA GPU
- CUDA 12+

If CUDA is available, prefer `--backend mjwp` for Stage 3. The CPU backend is always available as a fallback.

---

## Hardware setup

### Starting the Manus client and pose estimation

A convenience script launches both services in separate terminals:

```bash
bash setup.sh
```

This starts the Manus SDK client (integrated mode, auto-confirms the mode prompt) and the ArUco pose estimation server. Both must be running before any recording or live teleop script. `setup.sh` is the standard way to bring up the hardware stack at the start of a session.

### Tag pose estimation (one-time setup)

Camera and board calibration only needs to be redone when the physical setup changes. All commands run from `src/tag-pose-estimation/`.

```bash
# 1. Intrinsic calibration — run once per camera
python scripts/camera_intrinsic_calibration.py \
  --cam_name /dev/video6 \
  --charuco_board_path config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json

# 2. Build object boards (run once per physical board)
python scripts/build_object_board.py \
  --marker_size 0.05 --reference_marker <id> --name cube_7cm

# 3. Extrinsic calibration
python scripts/calibrate_extrinsics_dual_camera.py \
  --camera_1_config_path config/camera_config/realsense_1_config.json \
  --camera_2_config_path config/camera_config/realsense_2_config.json \
  --board_config_path config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json
```

### Manus Quantum Gloves

1. Connect the license USB dongle and the Bluetooth receiver.
2. Power on the gloves (steady blue LED = connected).
3. Run `bash setup.sh` (see above) — this handles the Manus client startup.
4. For a new user, follow the [Manus calibration guide](https://docs.manus-meta.com/2.4.0/Products/XR%20Metagloves/Calibration/).

---

## Calibration

Calibration maps each glove's wrist joint to its tracking board. Run these steps once per physical setup change.

### Coarse wrist-to-board transform

Point the fingertips at known world-frame positions with the glove flat:

```bash
python -m scripts.calibration.estimate_wrist_to_board
```

Results saved to `src/tag-pose-estimation/config/calibration_gloves/`.

### Fine hand-object alignment

With fingertips placed on the cube faces, optimises the calibration so tracked tips lie on the cube surface:

```bash
python -m scripts.calibration.optimize_hand_object_alignment
```

### Wrist offset calibration

Optimises the translational offset between the glove board and the robot wrist joint. The resulting constant (`DEFAULT_WRIST_OFFSET`) lives in `scripts/retargeting/tesollo_calibration.py` and is loaded automatically by all downstream scripts.

```bash
python -m scripts.calibration.optimize_wrist_offset
```

### Verify calibration quality

```bash
python -m scripts.evaluation.compute_calibration_metrics recordings/statics
```

Prints per-finger signed distance to the nearest cube face (target: < 3 mm).

---

## Main workflows

### Stage 1 — Capture a motion

```bash
# Record a new session and preprocess in one step
python -m scripts.capture_motion

# Reprocess an already-recorded session
python -m scripts.capture_motion recordings/<session>

# Force reprocessing
python -m scripts.capture_motion recordings/<session> --force

# Process and inspect the result
python -m scripts.capture_motion recordings/<session> --inspect

# Inspect the raw streams without reprocessing
python -m scripts.capture_motion recordings/<session> --inspect-raw
```

Recording requires the Manus client and pose estimation server to be running (`bash setup.sh`).

**What it does:**
1. `record_session` — streams `hands.jsonl`, `ergonomics.jsonl`, `poses.jsonl` to `recordings/session_YYYYMMDD_HHMMSS/`
2. `build_sequence_dataset` — resamples to 20 Hz, filters, rejects outliers, computes wrist transforms → `processed_sequence.npz`
3. `compute_proxy_geometry` — adds fingertip positions and directions in wrist/world/cube frames → `processed_sequence_with_proxy.npz`

---

### Stages 2–3 — Retarget a motion

```bash
# Full 3-stage retargeting pipeline (CPU backend)
python -m scripts.retarget_motion recordings/<session>

# GPU backend — much faster for Stage 3, requires CUDA 12+
python -m scripts.retarget_motion recordings/<session> --backend mjwp

# Force re-run all stages
python -m scripts.retarget_motion recordings/<session> --force

# Tagged run (does not overwrite canonical outputs)
python -m scripts.retarget_motion recordings/<session> --tag my_run

# Set MPPI random seed
python -m scripts.retarget_motion recordings/<session> --seed 42

# Skip stages for debugging (e.g. contacts already done, re-run Stage 3 only)
python -m scripts.retarget_motion recordings/<session> --skip-kinematic
python -m scripts.retarget_motion recordings/<session> --skip-kinematic --skip-contacts

# Run and open the MuJoCo viewer when done
python -m scripts.retarget_motion recordings/<session> --inspect
```

**What it does:**

| Stage | Script | Input | Output |
|---|---|---|---|
| 2 | `kinematic_retargeting` | `processed_sequence_with_proxy.npz` | `retargeted_sequence[__<tag>].npz` |
| 2.5 | `detect_contacts` | `retargeted_sequence.npz` | `contact_sequence[__<tag>].npz` |
| 3 | `optimize_sequence_dynamic` | both above | `retargeted_sequence_dynamic[__<tag>].npz` |

The `--tag` flag consistently suffixes all three output files so parameter-sweep runs never overwrite the canonical ones. For MPPI tuning knobs (`--horizon`, `--num-samples`, `--iters`, `--w-*`, …), call `optimize_sequence_dynamic` directly.

---

## Sub-step details

Each stage can be run in isolation for debugging.

### Stage 2 — Kinematic retargeting

```bash
python -m scripts.retargeting.kinematic_retargeting recordings/<session>
python -m scripts.retargeting.kinematic_retargeting recordings/<session> --force
python -m scripts.retargeting.kinematic_retargeting recordings/<session> --viewer
python -m scripts.retargeting.kinematic_retargeting recordings/<session> --tag cube_shift \
    --cube-shift 0.02 0 -0.01
```

SPIDER equality-constraint IK — maps each frame of the processed sequence to a robot joint configuration. The `--viewer` flag opens the MuJoCo passive viewer while solving.

**Key flags:**

| Flag | Description |
|---|---|
| `--force` | Overwrite existing output |
| `--viewer` | Show MuJoCo viewer during IK |
| `--tag TAG` | Write to `retargeted_sequence__TAG.npz` |
| `--cube-shift X Y Z` | Shift cube target in wrist frame (m) |
| `--cube-edge M` | Override cube edge length (m) |
| `--pin-wrist` | Lock wrist to reference pose |

---

### Stage 2.5 — Contact detection

```bash
python -m scripts.retargeting.detect_contacts recordings/<session>
python -m scripts.retargeting.detect_contacts recordings/<session> --force --stats
python -m scripts.retargeting.detect_contacts recordings/<session> \
    --stage2-tag my_run --tag my_run
```

Physics replay in MuJoCo that detects tip-cube collisions at each Stage 2 frame, then segments continuous contact intervals into a per-finger contact schedule.

**Key flags:**

| Flag | Description |
|---|---|
| `--force` | Overwrite existing output |
| `--stage2-tag TAG` | Read from `retargeted_sequence__TAG.npz` |
| `--tag TAG` | Write to `contact_sequence__TAG.npz` |
| `--sd-threshold M` | Signed-distance threshold for contact mask (m) |
| `--stats` | Print per-finger contact statistics |

---

### Stage 3 — Dynamic optimisation (MPPI)

```bash
python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session>
python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session> --force --seed 42
python -m scripts.retargeting.optimize_sequence_dynamic recordings/<session> \
    --tag sweep_h8 --horizon 8 --num-samples 2048 --iters 32
```

MPPI sampling-based MPC that rolls out finger and wrist controls in full physics (cube as free joint, no virtual stick), optimising for cube tracking and contact quality.

**Key flags (selection):**

| Flag | Default | Description |
|---|---|---|
| `--force` | — | Overwrite existing output |
| `--seed N` | 0 | Random seed |
| `--tag TAG` | — | Write to `retargeted_sequence_dynamic__TAG.npz` |
| `--stage2-tag TAG` | — | Read tagged Stage 2/2.5 inputs |
| `--horizon N` | 4 | MPPI planning horizon (steps) |
| `--num-samples N` | 1024 | MPPI sample count |
| `--iters N` | 16 | MPPI iterations per tick |
| `--num-knots N` | — | Spline knots (default: horizon) |
| `--substeps N` | 40 | Physics substeps per MPC step |
| `--sim-dt S` | 0.005 | Simulation timestep (s) |
| `--cube-edge M` | — | Override cube edge length (m) |
| `--cube-mass KG` | — | Override cube mass (kg) |
| `--w-cube-pos` | — | Cube position reward weight |
| `--w-cube-quat` | — | Cube rotation reward weight |
| `--start-time S` | 0.0 | Skip to this reference time (s) |
| `--max-ticks N` | 0 | Limit tick count (0 = full) |
| `--diagnostics` | — | Print per-term reward diagnostics |
| `--backend` | cpu | Rollout backend: `cpu` or `mjwp` (NVIDIA Warp GPU, requires CUDA 12+) |

---

## Visualization

All visualization scripts are in `scripts/visualization/`.

### Stage 1 — Capture inspection

```bash
# Interactive matplotlib animation of the processed session
python -m scripts.visualization.inspect_capture recordings/<session>
python -m scripts.visualization.inspect_capture recordings/<session> --raw   # raw streams
python -m scripts.visualization.inspect_capture recordings/<session> --speed 2.0

# Static diagnostic plots — raw streams (timestamps, rates, pose trajectories)
python -m scripts.visualization.inspect_capture_raw recordings/<session>

# Static diagnostic plots — processed sequence (trajectories, speeds, jumps)
python -m scripts.visualization.inspect_capture_processed recordings/<session>
```

### Stages 2–3 — Retargeting inspection

```bash
# Replay Stage 3 dynamic output in MuJoCo (orange ghost = MPC reference cube)
python -m scripts.visualization.inspect_retargeting recordings/<session>
python -m scripts.visualization.inspect_retargeting recordings/<session> --speed 0.5
python -m scripts.visualization.inspect_retargeting recordings/<session> --no-ref
python -m scripts.visualization.inspect_retargeting recordings/<session> --tag my_run

# Replay Stage 2 kinematic output
python -m scripts.visualization.inspect_retargeting recordings/<session> --stage2

# Preview a cube-shift in Stage 2 (green ghost = shifted target)
python -m scripts.visualization.inspect_retargeting recordings/<session> --stage2 \
    --cube-shift 0.02 0 -0.01
```

### Contact schedule

```bash
# Per-finger contact schedule plot (face assignment, contact mask, signed distance)
python -m scripts.visualization.plot_contact_schedule recordings/<session>
```

### Live teleop (require `bash setup.sh` first)

```bash
# Hand skeleton in world frame (matplotlib, no robot)
python -m scripts.visualization.live_skeleton_world

# Hand skeleton in wrist frame (matplotlib, no robot)
python -m scripts.visualization.live_skeleton_wrist

# IK retargeting to Tesollo — world frame (MuJoCo viewer)
python -m scripts.visualization.live_teleop_world

# IK retargeting to Tesollo — wrist frame (MuJoCo viewer)
python -m scripts.visualization.live_teleop_wrist

# IK retargeting to Tesollo — cube frame (MuJoCo viewer)
# --offset overrides the DEFAULT_WRIST_OFFSET (XYZ in metres, wrist-frame
# translation from the tracked board to the robot wrist joint)
python -m scripts.visualization.live_teleop_cube
python -m scripts.visualization.live_teleop_cube --offset 0.004 0.006 -0.027
```

---

## Evaluation

### Single recording

Computes all retargeting metrics for one session and shows diagnostic plots:

```bash
python -m scripts.evaluation.evaluate_recording recordings/<session>
python -m scripts.evaluation.evaluate_recording recordings/<session> --tag my_run
```

Prints a metric table and shows three plots: rotation error per tick, position error per tick, and off-axis wobble vs reference.

**Metrics:**

| Metric | Description |
|---|---|
| `success` | Completed without any tick exceeding 100 mm position error |
| `rot_mean / rot_max` | Mean / max rotation error (deg) |
| `pos_mean / pos_max` | Mean / max position error (mm) |
| `C_par` | Along-axis rotation coverage (deg) |
| `C_perp` | Off-axis / wobble coverage (deg) |
| `fin_geo` | Final geodesic error (deg) |
| `wall_s` | Wall-clock runtime (s) |

### Sweep analysis (thesis results)

Loads results from multiple sweep runs and generates comparison figures:

```bash
python -m scripts.evaluation.analyze_sweeps
```

Reads from the paths configured at the top of the script. Produces `documents/figures/`.

### Calibration quality

```bash
python -m scripts.evaluation.compute_calibration_metrics recordings/statics
```

---

## Script reference

### Top-level orchestrators

| Script | Purpose |
|---|---|
| `scripts/capture_motion.py` | Stage 1: record + preprocess a session |
| `scripts/retarget_motion.py` | Stages 2–3: full retargeting pipeline |

### `scripts/processing/`

| Script | Purpose |
|---|---|
| `record_session.py` | Record raw JSONL streams from Manus + pose estimator |
| `build_sequence_dataset.py` | Resample, filter, outlier-reject, compute wrist transforms |
| `compute_proxy_geometry.py` | Add fingertip positions/directions in wrist/world/cube frames |

### `scripts/retargeting/`

| Script | Purpose |
|---|---|
| `kinematic_retargeting.py` | Stage 2: SPIDER equality-constraint IK |
| `detect_contacts.py` | Stage 2.5: physics-replay contact detection |
| `optimize_sequence_dynamic.py` | Stage 3: MPPI dynamic optimisation |
| `build_ik_scene.py` | Build MuJoCo XML for the IK scene |
| `tesollo_calibration.py` | Wrist offset calibration constant (single source of truth) |
| `tesollo_target_mapping.py` | Maps human joint angles to robot targets |
| `tesollo_mink_ik.py` | Mink-based IK solver (used by live teleop and calibration) |
| `dynamic_retargeting/` | MPPI core: `mppi.py`, `cpu_env.py`, `mjwp_env.py`, `reward.py` |

### `scripts/visualization/`

| Script | Purpose |
|---|---|
| `inspect_capture.py` | Animated matplotlib view of a session (raw or processed) |
| `inspect_capture_raw.py` | Static diagnostic plots from raw JSONL streams |
| `inspect_capture_processed.py` | Static diagnostic plots from processed sequence |
| `inspect_retargeting.py` | MuJoCo viewer for Stage 2 or Stage 3 output |
| `plot_contact_schedule.py` | Per-finger contact schedule visualization |
| `live_skeleton_world.py` | Live Manus hand skeleton, world frame |
| `live_skeleton_wrist.py` | Live Manus hand skeleton, wrist frame |
| `live_teleop_world.py` | Live IK retargeting, world frame |
| `live_teleop_wrist.py` | Live IK retargeting, wrist frame |
| `live_teleop_cube.py` | Live IK retargeting, cube frame (optional `--offset` for wrist offset) |

### `scripts/calibration/`

| Script | Purpose |
|---|---|
| `estimate_wrist_to_board.py` | Coarse wrist-to-board transform from known fingertip positions |
| `optimize_hand_object_alignment.py` | Fine-tune calibration using cube surface contacts |
| `optimize_wrist_offset.py` | Optimise wrist position offset |

### `scripts/evaluation/`

| Script | Purpose |
|---|---|
| `evaluate_recording.py` | Metrics + plots for a single retargeted session |
| `analyze_sweeps.py` | Multi-run sweep analysis (thesis results) |
| `metrics.py` | Shared metric helpers (imported by both above) |
| `compute_calibration_metrics.py` | Calibration quality: fingertip-to-cube-surface error |

### `scripts/streams/`

| Script | Purpose |
|---|---|
| `live_inputs.py` | Unified ZMQ subscriber for all live data streams |
| `manus_skeleton_receiver.py` | Manus hand skeleton + fingertip receiver |
| `manus_ergonomics_receiver.py` | Manus ergonomics receiver |
| `board_pose_listener.py` | ArUco board pose subscriber |
| `image_listener.py` | Camera image subscriber |

### `scripts/utils/`

| Script | Purpose |
|---|---|
| `config.py` | Global constants (ports, board IDs, cube size, …) |
| `transforms.py` | SE(3) utility functions |

---

## Output file structure

After a complete run, a session directory contains:

```
recordings/session_YYYYMMDD_HHMMSS/
├── hands.jsonl                              # raw Manus hand stream
├── ergonomics.jsonl                         # raw Manus ergonomics stream
├── poses.jsonl                              # raw ArUco pose stream
├── meta.json                                # recording metadata
├── processed_sequence.npz                  # Stage 1: resampled + filtered timeline
├── processed_sequence_with_proxy.npz       # Stage 1: + fingertip proxy geometry
├── retargeted_sequence.npz                 # Stage 2: kinematic IK
├── contact_sequence.npz                    # Stage 2.5: contact schedule
├── retargeted_sequence_dynamic.npz         # Stage 3: MPPI output (canonical)
└── retargeted_sequence_dynamic__<tag>.npz  # Stage 3: tagged sweep runs
```

**Key arrays in `retargeted_sequence_dynamic.npz`:**

| Array | Shape | Description |
|---|---|---|
| `timeline` | `(N,)` | Wall-clock timestamps (s) |
| `retargeted_qpos` | `(N, 27)` | Hand joint angles at each tick |
| `cube_traj` | `(N, 7)` | Simulated cube pose (xyz + wxyz) |
| `cube_ref_used` | `(N, 7)` | MPC reference cube pose at each tick |
| `pos_err_history` | `(N,)` | Cube position error (m) |
| `rot_err_history` | `(N,)` | Cube rotation error (rad) |
| `run_config` | scalar JSON | All MPPI parameters used |
