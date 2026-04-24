# Tesollo Robohand — Teleoperation & Sequence Replay Pipeline

End-to-end pipeline for capturing hand-object interactions with Manus Quantum Gloves and ArUco tag-based pose estimation, retargeting them to the Tesollo robotic hand via an IK solver, and replaying the sequences in a MuJoCo simulation.

---

## Table of contents

1. [System overview](#system-overview)
2. [Installation](#installation)
3. [Hardware setup](#hardware-setup)
   - [Tag pose estimation](#tag-pose-estimation)
   - [Manus Quantum Gloves](#manus-quantum-gloves)
4. [Calibration](#calibration)
5. [Live teleoperation](#live-teleoperation)
6. [Recording & processing](#recording--processing)
7. [Offline retargeting](#offline-retargeting)
8. [Evaluation](#evaluation)
9. [Script reference](#script-reference)

---

## System overview

```
Manus gloves ──► hand skeleton + ergonomics (ZMQ)  ─┐
                                                      ├─► LiveInputs ──► retargeting (IK) ──► MuJoCo viewer
ArUco cameras ──► board + cube poses (ZMQ) ──────────┘

Recording:
  Raw JSONL streams ──► build_sequence_dataset ──► compute_proxy_geometry
                                                         │
                                          retarget_sequence_baseline
                                                         │
                                          replay in MuJoCo / save video
```

The world frame is defined by a fixed ArUco board on the table.  The left and right glove boards are tracked relative to this frame, and a calibration matrix maps each glove board to the wrist joint of the corresponding hand.  A 6 cm cube with an ArUco board is tracked as the manipulation object.

---

## Installation

```bash
git clone <this-repo>
cd robohand2
git submodule update --init --recursive
uv sync
```

---

## Hardware setup

### Tag pose estimation

All commands below are run from inside `src/tag-pose-estimation/`.

#### 1 — Intrinsic camera calibration

Run once per camera.  Hold a ChArUco board in front of the camera at various angles and distances; press `s` to capture a frame, `q` when done.

```bash
python scripts/camera_intrinsic_calibration.py \
  --cam_name /dev/video6 \
  --charuco_board_path config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json
```

Use `--focus <value>` if autofocus needs to be locked.  Results are saved into the camera config file.

#### 2 — Build object boards

Each tracked object (left glove, right glove, cube) needs a board config that describes the 3D layout of its ArUco markers.  Place the object flat, run pose estimation briefly to detect the markers, then build the board:

```bash
python scripts/build_object_board.py \
  --marker_size 0.05 \
  --reference_marker <anchor_marker_id> \
  --name cube_7cm
```

Existing board configs are in `config/object_boards/` (e.g. `cube_7cm.json`, `hand_tracking_left.json`).

#### 3 — Visualise a board

Sanity-check the marker layout of any board config before using it:

```bash
python scripts/visualise_board.py \
  --filepath config/object_boards/cube_7cm.json
```

#### 4a — Extrinsic calibration: single camera

Hold a ChArUco board in view and move it around.  Establishes the camera-to-world transform.

```bash
python scripts/calibrate_extrinsics_single_camera.py \
  --camera_config_path config/camera_config/realsense_1_config.json \
  --board_config_path config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json
```

#### 4b — Extrinsic calibration: dual camera

Calibrates two cameras jointly so they share a common world frame.

```bash
python scripts/calibrate_extrinsics_dual_camera.py \
  --camera_1_config_path config/camera_config/realsense_1_config.json \
  --camera_2_config_path config/camera_config/realsense_2_config.json \
  --board_config_path config/calibration_boards/default_intrinsic_calibration_board/charuco_board.json
```

#### 5 — Run pose estimation

Starts the pose estimator, opens live camera streams, and publishes board poses over ZMQ.  This must be running before any teleop or recording script.

```bash
python scripts/run_pose_estimation.py \
  --config config/pose_estimation_configs/manus_pose_estimation_config.json
```

The config file controls which cameras, which boards to track, the output ZMQ address, and whether to stream images.  Current setup: two RealSense cameras, ~15 Hz, three boards tracked (left glove, right glove, cube).

### Manus Quantum Gloves

1. Connect the license USB dongle and the Bluetooth receiver.
2. Power on the gloves.  Steady blue LED = connected.
3. Start the Manus client in integrated mode:
   ```bash
   # in src/manus/LinuxClient
   ./ManusCoreConsumer    # then press 1 for integrated mode
   ```
4. Verify streaming by running a teleop script (see below).
5. For a new user, redo the hand calibration following the [Manus calibration guide](https://docs.manus-meta.com/2.4.0/Products/XR%20Metagloves/Calibration/).

---

## Calibration

Calibration maps each glove's wrist joint to its ArUco tracking board.  There are two stages:

### Stage 1 — Coarse wrist-to-board calibration

Point the fingertips at known world-frame positions with the glove flat.  The script fits a rigid transform from wrist frame to board frame.

```bash
python -m scripts.calibration.estimate_wrist_to_board
```

Results are saved to `src/tag-pose-estimation/config/calibration_gloves/`.

### Stage 2 — Fine hand-object alignment optimisation

With the fingertips placed on the faces of the cube, this script optimises the calibration matrix so that the tracked fingertips lie on the cube surface.  Produces `T_board_wrist_optimized`, which is loaded automatically by all downstream scripts.

```bash
python -m scripts.calibration.optimize_hand_object_alignment
```

### Verify calibration quality

```bash
python -m scripts.evaluation.compute_calibration_metrics recordings/statics
```

Prints per-finger signed distance to the nearest cube face (should be close to 0 mm) and a bar chart of the error distribution.

---

## Live teleoperation

All scripts below require the Manus client and pose estimation to be running first.

### Skeleton visualisation (matplotlib, no robot)

Plots the hand skeleton and cube in real time — useful for verifying streams before running the IK.

```bash
# World frame (hands + cube relative to table)
python -m scripts.apps.run_manus_teleop_world

# Wrist frame (hand relative to glove board)
python -m scripts.apps.run_manus_teleop_wrist
```

### IK retargeting to Tesollo hand (MuJoCo viewer)

Runs the full inverse-kinematics pipeline and shows the retargeted robot hand in the MuJoCo passive viewer.

```bash
# World frame — hand tracks the physical wrist position in space
python -m scripts.apps.run_tesollo_teleop_world

# Wrist frame — hand tracks finger configuration only, ignores wrist pose
python -m scripts.apps.run_tesollo_teleop_wrist
```

---

## Recording & processing

### 1 — Record a session

Records all streams (hand skeleton, ergonomics, board + cube poses) to timestamped JSONL files under `recordings/<session>/`.

```bash
python -m scripts.apps.record_demo_sequence
```

Output: `recordings/session_YYYYMMDD_HHMMSS/{hands,ergonomics,poses}.jsonl`

### 2 — Inspect raw data

Plots raw stream timestamps, arrival rates, and pose trajectories directly from the JSONL files — useful for diagnosing recording issues before processing.

```bash
python -m scripts.apps.replay_sequence recordings/<session> --raw

# Or use the dedicated inspector for detailed diagnostics
python -m scripts.processing.inspect_raw_sequence recordings/<session>
```

### 3 — Process the recording

The processing pipeline resamples all streams to a uniform 20 Hz timeline, rejects pose outliers (false ArUco detections), applies low-pass filtering, and computes world-frame transforms.

**Orchestrator — runs all steps, skips those already done:**

```bash
# Process a single session
python -m scripts.apps.process_recordings recordings/<session>

# Process all sessions found under recordings/
python -m scripts.apps.process_recordings

# Force re-run even if outputs exist
python -m scripts.apps.process_recordings recordings/<session> --force

# Process then open the processed inspector
python -m scripts.apps.process_recordings recordings/<session> --inspect

# Open the raw inspector without (re)processing
python -m scripts.apps.process_recordings recordings/<session> --inspect-raw
```

**Individual steps:**

```bash
# Step 1: resample, filter, compute wrist transforms
python -m scripts.processing.build_sequence_dataset recordings/<session>

# Step 2: derive fingertip positions and directions in wrist, world, and cube frames
python -m scripts.processing.compute_proxy_geometry recordings/<session>/processed_sequence.npz
```

Output:
- `processed_sequence.npz` — uniform timeline, filtered poses, wrist transforms, joints, ergonomics
- `processed_sequence_with_proxy.npz` — adds fingertip positions/directions in all frames

### 4 — Inspect processed data

```bash
# Animated 3D plot of hand skeleton + cube (loops)
python -m scripts.apps.replay_sequence recordings/<session>
python -m scripts.apps.replay_sequence recordings/<session> --speed 2.0

# Static diagnostic plots (trajectories, speeds, orientation jumps)
python -m scripts.processing.inspect_processed_sequence recordings/<session>
```

### Processing parameters

Key constants at the top of `scripts/processing/build_sequence_dataset.py`:

| Parameter | Default | Description |
|---|---|---|
| `TARGET_HZ` | 20 Hz | Resampling frequency |
| `OUTLIER_MAX_POS_JUMP_M` | 0.06 m | Reject pose frames jumping more than this |
| `OUTLIER_MAX_ROT_JUMP_DEG` | 20° | Reject pose frames rotating more than this |
| `FILTER_HZ_POSES` | 3 Hz | Low-pass cutoff for board/cube poses |
| `FILTER_HZ_HANDS` | 8 Hz | Low-pass cutoff for hand joints and ergonomics |

---

## Offline retargeting

Processes a recorded session through the same IK pipeline as live teleop, producing retargeted robot joint angles for every frame.

### Run retargeting

```bash
# Retarget and save (headless, fastest)
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session>

# Retarget with live MuJoCo viewer
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session> --viewer

# Retarget and immediately save a video
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session> --save-video

# Force re-run if output already exists
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session> --force
```

Output: `recordings/<session>/retargeted_sequence_baseline.npz`  
Keys: `timeline (N,)`, `retargeted_qpos (N, nq)`

### Replay retargeted sequence

```bash
# Interactive MuJoCo viewer (loops)
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session> --replay

# Replay and render to MP4 (requires: pip install imageio[ffmpeg])
python -m scripts.retargeting.retarget_sequence_baseline recordings/<session> --replay --save-video
```

Output: `recordings/<session>/retargeted_sequence_baseline.mp4`

---

## Evaluation

### Retargeting quality

Compares robot fingertip positions (from forward kinematics) to human fingertip positions, both expressed in the cube frame.  Per-finger position error in mm.

```bash
# Single session
python -m scripts.evaluation.compute_retargeting_metrics recordings/<session>

# All sessions that have a retargeted file
python -m scripts.evaluation.compute_retargeting_metrics recordings/
```

Plots: per-finger error time series + mean ± std bar chart.

### Calibration quality (static poses)

Measures how well the calibrated fingertip positions lie on the cube surface.

```bash
python -m scripts.evaluation.compute_calibration_metrics recordings/statics
```

---

## Script reference

### `scripts/apps/` — runnable applications

| Script | Purpose |
|---|---|
| `record_demo_sequence.py` | Record a session to JSONL files |
| `process_recordings.py` | Orchestrate the full processing pipeline |
| `replay_sequence.py` | Animate a session in matplotlib (`--raw` for unprocessed) |
| `run_manus_teleop_world.py` | Live hand skeleton visualisation, world frame |
| `run_manus_teleop_wrist.py` | Live hand skeleton visualisation, wrist frame |
| `run_tesollo_teleop_world.py` | Live IK retargeting, world frame, MuJoCo viewer |
| `run_tesollo_teleop_wrist.py` | Live IK retargeting, wrist frame, MuJoCo viewer |

### `scripts/calibration/` — calibration

| Script | Purpose |
|---|---|
| `estimate_wrist_to_board.py` | Coarse wrist-to-board transform from known finger positions |
| `optimize_hand_object_alignment.py` | Fine-tune calibration using cube surface contacts |

### `scripts/processing/` — data processing

| Script | Purpose |
|---|---|
| `build_sequence_dataset.py` | Resample, filter, outlier-reject, compute wrist transforms |
| `compute_proxy_geometry.py` | Add fingertip positions/directions in all frames |
| `inspect_raw_sequence.py` | Diagnostic plots from raw JSONL streams |
| `inspect_processed_sequence.py` | Diagnostic plots from processed sequence |

### `scripts/retargeting/` — offline retargeting

| Script | Purpose |
|---|---|
| `retarget_sequence_baseline.py` | IK retargeting of a recorded session; replay and video export |

### `scripts/evaluation/` — metrics

| Script | Purpose |
|---|---|
| `compute_calibration_metrics.py` | Fingertip-to-cube-surface distance for static poses |
| `compute_retargeting_metrics.py` | Robot vs. human fingertip error in cube frame |
| `compute_simulation_metrics.py` | *(placeholder)* |
