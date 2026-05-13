"""
Offline retargeting using cube-relative fingertip targets.

IK targets are placed at human fingertip positions expressed in cube frame,
then transformed to world frame via the tracked cube pose.  This decouples
targets from wrist-tracking noise and keeps them on/near the cube surface,
which sets up clean contact geometry for the downstream SPIDER physics step.

When the cube is not visible (cube_pose_world[i] is None) the frame is
skipped and a NaN row is written — consistent with the baseline behaviour for
invalid frames.

Modes
-----
(default)      retarget and save retargeted_sequence_cube_relative.npz
--viewer       show interactive viewer while retargeting
--replay       load saved .npz and replay in viewer (no re-computation)
--save-video   render to retargeted_sequence_cube_relative.mp4
--force        overwrite existing output

Inputs:  recordings/<session>/processed_sequence_with_proxy.npz
Output:  recordings/<session>/retargeted_sequence_cube_relative.npz
           Keys: timeline (N,), retargeted_qpos (N, nq)
         recordings/<session>/retargeted_sequence_cube_relative.mp4  (--save-video)

Usage:
    python -m scripts.retargeting.retarget_sequence_cube_relative recordings/<session>
    python -m scripts.retargeting.retarget_sequence_cube_relative recordings/<session> --viewer
    python -m scripts.retargeting.retarget_sequence_cube_relative recordings/<session> --replay
    python -m scripts.retargeting.retarget_sequence_cube_relative recordings/<session> --force
"""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.tesollo_constants import FINGERS, TARGET_SCALE
from scripts.retargeting.tesollo_mujoco_utils import (
    initialize_fingertip_target_mocaps,
    set_hand_base_pose,
    set_mocap_body_pose_from_matrix,
)
from scripts.retargeting.tesollo_ik_runtime import (
    initialize_targets_from_configuration,
    reset_relative_finger_targets,
    solve_ik_step,
    update_finger_frame_tasks_from_mocaps,
    update_fingertip_targets_cube_frame,
    update_posture_target_from_joint_angles,
    update_relative_finger_targets,
    update_wrist_task_target,
)
from scripts.retargeting.tesollo_task_builder import build_tesollo_task_bundle
from scripts.retargeting.tesollo_target_mapping import _clamp_target_outside_cube
from scripts.utils.transforms import pose_to_matrix

MODEL_PATH = "/home/suno/robohand2/assets/output_mocap.xml"
IK_STEPS = 50
IK_DT = 0.002
PLAYBACK_HZ = 30.0
WARMUP_CALLS = 30

VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280

# Per-finger wrist-frame offsets, calibrated via
# scripts.calibration.optimize_finger_offsets against the statics holds.
# Closes the systematic glove-landmark vs robot-tip-geom gap. Interpreted in
# wrist frame (applied via map_cube_frame_target → rotated wrist→cube each
# frame), so calibration generalises across cube orientations.
OFFLINE_TARGET_OFFSETS = {
    "thumb_tip":  np.array([-0.00229, +0.00614, +0.00316]),
    "index_tip":  np.array([+0.00716, +0.00136, -0.00482]),
    "middle_tip": np.array([-0.00832, +0.00678, -0.00896]),
    "ring_tip":   np.array([-0.01872, +0.00496, -0.00579]),
    "little_tip": np.array([-0.00558, +0.02361, +0.00472]),
}

_IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])

# Wrist→robot-base translational offset, expressed in wrist frame (metres).
# Compensates for the offset between the Manus wrist landmark and the robot
# freejoint origin. Optimized via scripts/calibration/optimize_wrist_offset.py
# against the statics holds. Override via `wrist_offset` arg or `--offset tx ty tz`.
DEFAULT_WRIST_OFFSET = np.array([0.0043, 0.0064, -0.0268], dtype=float)

# Target-shift: translates the *cube* (and therefore all cube-relative fingertip
# targets) by this vector — expressed in wrist frame, in metres. Used to push
# the targets away from the palm so the robot's longer fingers don't have to
# curl as tightly. Wrist stays fixed; relative geometry preserved.
DEFAULT_TARGET_SHIFT = np.zeros(3, dtype=float)


def _apply_wrist_offset(T_world_wrist, offset):
    """Translate the wrist transform by `offset` expressed in the wrist frame."""
    if T_world_wrist is None:
        return None
    T = np.asarray(T_world_wrist, dtype=float).copy()
    T[:3, 3] = T[:3, :3] @ np.asarray(offset, dtype=float) + T[:3, 3]
    return T


def _shift_cube_pose(cube_pose7, T_world_wrist, shift):
    """Translate cube pose by `shift` expressed in wrist frame.

    cube_pose7 is [x,y,z, qw,qx,qy,qz]. Returns shifted pose, same format.
    """
    if cube_pose7 is None or T_world_wrist is None:
        return cube_pose7
    cp = np.asarray(cube_pose7, dtype=float).copy()
    shift_world = np.asarray(T_world_wrist, dtype=float)[:3, :3] @ np.asarray(shift, dtype=float)
    cp[:3] = cp[:3] + shift_world
    return cp


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_quats(n: int) -> np.ndarray:
    return np.tile(_IDENTITY_QUAT_XYZW, (n, 1))


def _load_processed(session_dir: Path):
    p = session_dir / "processed_sequence_with_proxy.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"processed_sequence_with_proxy.npz not found in {session_dir}.\n"
            "Run compute_proxy_geometry first."
        )
    return np.load(p, allow_pickle=True)


def _pin_wrist_in_configuration(task_bundle, T_world_wrist):
    T = np.asarray(T_world_wrist, dtype=float)
    pos = T[:3, 3]
    rot_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    q = task_bundle.configuration.q
    q[0:3] = pos
    q[3] = rot_xyzw[3]
    q[4] = rot_xyzw[0]
    q[5] = rot_xyzw[1]
    q[6] = rot_xyzw[2]



def _set_cube_mocap(model, data, cube_pose):
    if cube_pose is None:
        return
    try:
        T = pose_to_matrix(np.asarray(cube_pose, dtype=float))
        set_mocap_body_pose_from_matrix(model, data, "tracked_cube", T)
    except Exception:
        pass


def _make_camera(model) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 0.9
    cam.azimuth = 135.0
    cam.elevation = -25.0
    return cam


def _apply_qpos(model, data, qpos, cube_pose=None):
    data.qpos[:len(qpos)] = qpos
    _set_cube_mocap(model, data, cube_pose)
    mujoco.mj_forward(model, data)


# ── IK frame ──────────────────────────────────────────────────────────────────

def _ik_frame(
    model, mj_data, task_bundle, default_quats,
    fingertips_cube, T_world_cube, T_world_wrist, ergonomics,
    scale=TARGET_SCALE, wrist_offset=DEFAULT_WRIST_OFFSET,
):
    """One frame of cube-relative IK.

    `wrist_offset` is a translation in wrist-frame coords applied to the robot
    freejoint pose. Targets remain anchored to the cube; only the robot base
    shifts.

    Returns configuration.q if targets are valid (cube visible + wrist tracked),
    otherwise returns NaN-filled array.
    """
    T_world_robot_base = _apply_wrist_offset(T_world_wrist, wrist_offset)

    _set_cube_mocap(model, mj_data, None if T_world_cube is None else
                    _matrix_to_pose(T_world_cube))

    has_targets = update_fingertip_targets_cube_frame(
        model=model,
        data=mj_data,
        task_bundle=task_bundle,
        fingertip_positions_cube=fingertips_cube,
        fingertip_quats=default_quats,
        T_world_from_cube=T_world_cube,
        T_world_from_wrist=T_world_robot_base,  # rotation only used; offset is pure translation
        offsets=OFFLINE_TARGET_OFFSETS,
        scale=scale,
    )

    if has_targets:
        update_relative_finger_targets(task_bundle)
    else:
        reset_relative_finger_targets(task_bundle)

    update_posture_target_from_joint_angles(task_bundle, ergonomics, T_world_robot_base)
    update_finger_frame_tasks_from_mocaps(model, mj_data, task_bundle)
    update_wrist_task_target(task_bundle, T_world_robot_base)

    for _ in range(IK_STEPS):
        solve_ik_step(task_bundle, dt=IK_DT)

    # Sync mj_data.qpos with the IK-determined configuration so downstream
    # FK / mocap reads see the result.
    mj_data.qpos[: task_bundle.configuration.q.shape[0]] = task_bundle.configuration.q
    mujoco.mj_forward(model, mj_data)

    return task_bundle.configuration.q.copy()


def _matrix_to_pose(T):
    """4×4 matrix → [x,y,z,qw,qx,qy,qz] pose vector."""
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return np.array([pos[0], pos[1], pos[2],
                     quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


# ── retargeting ───────────────────────────────────────────────────────────────

def retarget_offline(
    session_dir: Path,
    show_viewer: bool = False,
    force: bool = False,
    output_name: str = "retargeted_sequence_cube_relative.npz",
    wrist_offset=DEFAULT_WRIST_OFFSET,
    target_shift=DEFAULT_TARGET_SHIFT,
):
    output_path = session_dir / output_name
    if output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (--force to overwrite)")
        return

    wrist_offset = np.asarray(wrist_offset, dtype=float)
    target_shift = np.asarray(target_shift, dtype=float)

    npz = _load_processed(session_dir)
    timeline = np.asarray(npz["timeline"], dtype=float)
    N = len(timeline)

    right_fingertips_cube = npz["right_fingertips_cube"]
    cube_pose_world       = npz["cube_pose_world"]
    right_wrist_world     = npz["right_wrist_world"]
    right_ergonomics      = npz["right_ergonomics"]

    # Apply target shift in wrist frame: cube + all cube-relative targets move
    # together by R_world_wrist @ target_shift. Wrist itself stays fixed.
    if np.any(target_shift != 0.0):
        cube_pose_used = np.empty(N, dtype=object)
        for i in range(N):
            cube_pose_used[i] = _shift_cube_pose(
                cube_pose_world[i], right_wrist_world[i], target_shift,
            )
    else:
        cube_pose_used = cube_pose_world

    # Build T_world_cube matrices from (possibly shifted) pose vectors
    T_world_cube_arr = np.empty(N, dtype=object)
    for i in range(N):
        if cube_pose_used[i] is not None:
            T_world_cube_arr[i] = pose_to_matrix(np.asarray(cube_pose_used[i], dtype=float))

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    task_bundle = build_tesollo_task_bundle(model)
    model = task_bundle.configuration.model
    mj_data = task_bundle.configuration.data

    initialize_targets_from_configuration(task_bundle)
    initialize_fingertip_target_mocaps(model, mj_data, task_bundle.fingers)

    default_quats = _default_quats(len(task_bundle.fingers))
    retargeted_qpos = np.full((N, model.nq), np.nan)

    # Warm up on the first frame where both cube and wrist are tracked.
    first_i = next(
        (i for i in range(N)
         if right_fingertips_cube[i] is not None and T_world_cube_arr[i] is not None
         and right_wrist_world[i] is not None),
        None,
    )
    if first_i is None:
        print("  [error] no frame with cube + wrist both tracked — cannot retarget")
        return

    print(f"  Warming up on frame {first_i} ({WARMUP_CALLS} × {IK_STEPS} IK steps)"
          f"   wrist_offset={wrist_offset.tolist()}")
    for _ in range(WARMUP_CALLS):
        _ik_frame(
            model, mj_data, task_bundle, default_quats,
            right_fingertips_cube[first_i], T_world_cube_arr[first_i],
            right_wrist_world[first_i], right_ergonomics[first_i],
            wrist_offset=wrist_offset,
        )

    print(f"  Retargeting {N} frames ({IK_STEPS} IK steps/frame, cube-relative) …")

    def _process_frame(i):
        return _ik_frame(
            model, mj_data, task_bundle, default_quats,
            right_fingertips_cube[i], T_world_cube_arr[i],
            right_wrist_world[i], right_ergonomics[i],
            wrist_offset=wrist_offset,
        )

    if show_viewer:
        with mujoco.viewer.launch_passive(
            model=model, data=mj_data, show_left_ui=False, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            rate = RateLimiter(frequency=PLAYBACK_HZ, warn=False)
            for i in range(N):
                if not viewer.is_running():
                    break
                retargeted_qpos[i] = _process_frame(i)
                mujoco.mj_camlight(model, mj_data)
                viewer.sync()
                rate.sleep()
                if (i + 1) % 200 == 0:
                    print(f"    [{i + 1}/{N}]")
    else:
        for i in range(N):
            retargeted_qpos[i] = _process_frame(i)
            if (i + 1) % 200 == 0:
                print(f"    [{i + 1}/{N}]")

    # Stack the (possibly shifted) cube poses we actually targeted
    cube_pose_used_arr = np.full((N, 7), np.nan)
    for i in range(N):
        if cube_pose_used[i] is not None:
            cube_pose_used_arr[i] = np.asarray(cube_pose_used[i], dtype=float)

    np.savez_compressed(
        output_path,
        timeline=timeline,
        retargeted_qpos=retargeted_qpos,
        wrist_offset=wrist_offset,
        target_shift=target_shift,
        cube_pose_used=cube_pose_used_arr,
    )
    print(f"  Saved: {output_path}")


# ── replay ────────────────────────────────────────────────────────────────────

def _load_retargeted(session_dir: Path, name: str):
    path = session_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}\nRun without --replay first.")
    return np.load(path)


def _load_cube_poses(session_dir: Path, N: int):
    p = session_dir / "processed_sequence_with_proxy.npz"
    if p.exists():
        return np.load(p, allow_pickle=True).get(
            "cube_pose_world", np.array([None] * N, dtype=object)
        )
    return np.array([None] * N, dtype=object)


def _load_target_inputs(session_dir: Path, N: int):
    """Load cube poses, cube-frame fingertip positions, and wrist world poses
    (the wrist rotation is needed to apply wrist-frame OFFLINE_TARGET_OFFSETS,
    matching how the IK applies them in map_cube_frame_target)."""
    p = session_dir / "processed_sequence_with_proxy.npz"
    if not p.exists():
        return None, None, None
    d = np.load(p, allow_pickle=True)
    return (d.get("cube_pose_world"),
            d.get("right_fingertips_cube"),
            d.get("right_wrist_world"))


def _update_target_mocaps(model, data, T_world_cube, fingertips_cube,
                          T_world_wrist=None, scale=TARGET_SCALE):
    """Update the {finger}_target mocap bodies. Mirrors map_cube_frame_target:
    OFFLINE_TARGET_OFFSETS are wrist-frame, rotated wrist→cube per frame."""
    if T_world_cube is None or fingertips_cube is None:
        return
    R_wc = T_world_cube[:3, :3]
    t_wc = T_world_cube[:3, 3]
    R_ww = T_world_wrist[:3, :3] if T_world_wrist is not None else None
    for i, finger in enumerate(FINGERS):
        offset_wrist = OFFLINE_TARGET_OFFSETS[finger]
        offset_cube = R_wc.T @ (R_ww @ offset_wrist) if R_ww is not None else offset_wrist
        pos_cube = np.asarray(fingertips_cube[i], dtype=float) * scale + offset_cube
        pos_cube = _clamp_target_outside_cube(pos_cube)
        pos_world = R_wc @ pos_cube + t_wc
        mocap_id = model.body(f"{finger}_target").mocapid[0]
        data.mocap_pos[mocap_id] = pos_world


def replay_sequence(session_dir: Path, save_video: bool = False):
    npz = _load_retargeted(session_dir, "retargeted_sequence_cube_relative.npz")
    timeline = np.asarray(npz["timeline"], dtype=float)
    retargeted_qpos = npz["retargeted_qpos"]
    N = len(timeline)
    # Prefer the shifted cube poses that were actually used for retargeting,
    # so visualization matches the IK targets.
    if "cube_pose_used" in npz.files:
        cube_pose_world = npz["cube_pose_used"]
    else:
        cube_pose_world = _load_cube_poses(session_dir, N)
    _, right_fingertips_cube, right_wrist_world = _load_target_inputs(session_dir, N)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if save_video:
        _render_video(session_dir, model, data, timeline, retargeted_qpos,
                      cube_pose_world, right_fingertips_cube, right_wrist_world)
    else:
        _replay_viewer(model, data, timeline, retargeted_qpos,
                       cube_pose_world, right_fingertips_cube, right_wrist_world)


def _replay_viewer(model, data, timeline, retargeted_qpos, cube_pose_world,
                   fingertips_cube=None, wrist_world=None):
    N = len(timeline)
    dt_mean = float(np.diff(timeline).mean()) if N > 1 else 1.0 / PLAYBACK_HZ
    fps = min(1.0 / dt_mean, PLAYBACK_HZ)
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        rate = RateLimiter(frequency=fps, warn=False)
        print(f"  Replaying {N} frames at {fps:.1f} Hz … (close viewer to stop)")
        while viewer.is_running():
            for i in range(N):
                if not viewer.is_running():
                    break
                _apply_qpos(model, data, retargeted_qpos[i], cube_pose_world[i])
                if fingertips_cube is not None and cube_pose_world[i] is not None:
                    T = pose_to_matrix(np.asarray(cube_pose_world[i], dtype=float))
                    Tw = (np.asarray(wrist_world[i], dtype=float)
                          if wrist_world is not None and wrist_world[i] is not None
                          else None)
                    _update_target_mocaps(model, data, T, fingertips_cube[i], Tw)
                    mujoco.mj_forward(model, data)
                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()


def _render_video(session_dir, model, data, timeline, retargeted_qpos, cube_pose_world,
                  fingertips_cube=None, wrist_world=None):
    try:
        import imageio
    except ImportError:
        print("  [error] imageio not found — pip install imageio[ffmpeg]")
        return

    N = len(timeline)
    dt_mean = float(np.diff(timeline).mean()) if N > 1 else 1.0 / PLAYBACK_HZ
    video_fps = max(1, round(1.0 / dt_mean))
    video_path = session_dir / "retargeted_sequence_cube_relative.mp4"

    model.vis.global_.offwidth = VIDEO_WIDTH
    model.vis.global_.offheight = VIDEO_HEIGHT
    renderer = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    cam = _make_camera(model)

    print(f"  Rendering {N} frames → {video_path.name}  (fps={video_fps}) …")
    with imageio.get_writer(str(video_path), fps=video_fps) as writer:
        for i in range(N):
            _apply_qpos(model, data, retargeted_qpos[i], cube_pose_world[i])
            if fingertips_cube is not None and cube_pose_world[i] is not None:
                T = pose_to_matrix(np.asarray(cube_pose_world[i], dtype=float))
                Tw = (np.asarray(wrist_world[i], dtype=float)
                      if wrist_world is not None and wrist_world[i] is not None
                      else None)
                _update_target_mocaps(model, data, T, fingertips_cube[i], Tw)
                mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            if (i + 1) % 200 == 0:
                print(f"    [{i + 1}/{N}]")

    renderer.close()
    print(f"  Saved video: {video_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_vec3(args, flag, default):
    """Parse `<flag> tx ty tz` from CLI args list (modifies args in place)."""
    if flag in args:
        i = args.index(flag)
        try:
            tx, ty, tz = float(args[i+1]), float(args[i+2]), float(args[i+3])
        except (IndexError, ValueError):
            print(f"Error: {flag} requires three floats: {flag} tx ty tz")
            sys.exit(1)
        del args[i:i+4]
        return np.array([tx, ty, tz], dtype=float)
    return default


def _parse_offset(args):
    return _parse_vec3(args, "--offset", DEFAULT_WRIST_OFFSET)


def _parse_target_shift(args):
    return _parse_vec3(args, "--target-shift", DEFAULT_TARGET_SHIFT)


def main():
    args = sys.argv[1:]
    show_viewer = "--viewer" in args
    do_replay   = "--replay" in args
    save_video  = "--save-video" in args
    force       = "--force" in args
    wrist_offset = _parse_offset(args)
    target_shift = _parse_target_shift(args)
    paths = [Path(a) for a in args if not a.startswith("--")]

    if not paths:
        print(
            "Usage: python -m scripts.retargeting.retarget_sequence_cube_relative <session_dir>"
            " [--viewer] [--replay] [--save-video] [--force]"
            " [--offset tx ty tz] [--target-shift tx ty tz]"
        )
        sys.exit(1)

    for p in paths:
        print(f"\n{p.name}")
        if do_replay:
            replay_sequence(p, save_video=save_video)
        else:
            retarget_offline(p, show_viewer=show_viewer, force=force,
                             wrist_offset=wrist_offset, target_shift=target_shift)
            if save_video:
                npz = _load_retargeted(p, "retargeted_sequence_cube_relative.npz")
                timeline = np.asarray(npz["timeline"], dtype=float)
                retargeted_qpos = npz["retargeted_qpos"]
                cube_pose_world = _load_cube_poses(p, len(timeline))
                _, fingertips_cube, wrist_world = _load_target_inputs(p, len(timeline))
                model = mujoco.MjModel.from_xml_path(MODEL_PATH)
                data = mujoco.MjData(model)
                _render_video(p, model, data, timeline, retargeted_qpos,
                              cube_pose_world, fingertips_cube, wrist_world)


if __name__ == "__main__":
    main()
