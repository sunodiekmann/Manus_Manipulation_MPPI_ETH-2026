"""SPIDER-style equality-constraint IK over a session.

For each recorded frame:
  * Wrist mocap pose = `T_world_wrist @ wrist_offset` (existing calibration).
  * Cube target mocap pose = recorded `cube_pose_world`.
  * Fingertip mocap poses = world-frame positions computed via
    `map_cube_frame_target` (anchored to cube, cube-relative offsets +
    wrist-frame calibration).
  * Run `mj_step` K = ref_dt / sim_dt times. The IK scene's equality
    springs (built in `build_ik_scene.py`) pull the hand toward the targets.
  * Pin the wrist qpos[0:7] each step — the recorded wrist pose is the
    desired pose; the freejoint shouldn't be driven by finger-spring
    reaction forces through the kinematic chain.

Frame 0 uses SPIDER's random-initial-guess search (8 candidates, best
mocap residual wins).

Outputs `recordings/<session>/retargeted_sequence.npz`:
    timeline         (N,)
    retargeted_qpos  (N, 27)   wrist freejoint + 20 finger joints
    cube_pose_used   (N, 7)    [x, y, z, qw, qx, qy, qz]
    wrist_offset     (3,)

CLI:
    python -m scripts.retargeting.retarget_sequence_spider recordings/<session>
    python -m scripts.retargeting.retarget_sequence_spider recordings/<session> --viewer
    python -m scripts.retargeting.retarget_sequence_spider recordings/<session> --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.build_ik_scene import (
    CUBE_TARGET_MOCAP_BODY,
    FINGER_TIPS,
    WRIST_MOCAP_BODY,
    compile_ik,
)
from scripts.retargeting.tesollo_calibration import (
    DEFAULT_WRIST_OFFSET,
    OFFLINE_TARGET_OFFSETS,
    apply_wrist_offset,
)
from scripts.retargeting.tesollo_target_mapping import map_cube_frame_target
from scripts.utils.transforms import pose_to_matrix


SIM_DT = 0.01
REF_DT = 1.0 / 30.0
STEPS_PER_FRAME = max(1, int(round(REF_DT / SIM_DT)))
NUM_INITIAL_GUESSES = 8
INITIAL_GUESS_STEPS = 30
PLAYBACK_HZ = 30.0


def _xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _T_to_mocap(T):
    """Convert a 4x4 transform to (pos, quat_wxyz)."""
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos.copy(), _xyzw_to_wxyz(quat_xyzw)


def _make_index(model):
    """Build a name → mocap_idx / site_id map for the bodies we touch."""
    info = {}
    for name in (WRIST_MOCAP_BODY, CUBE_TARGET_MOCAP_BODY,
                 *(ft.target_body for ft in FINGER_TIPS)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        info[name] = model.body_mocapid[bid]
    site_ids = {}
    for ft in FINGER_TIPS:
        site_ids[f"{ft.finger}_site"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{ft.finger}_site")
        site_ids[f"{ft.target_body}_site"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{ft.target_body}_site")
    site_ids["cube_site"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
    site_ids[f"{CUBE_TARGET_MOCAP_BODY}_site"] = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, f"{CUBE_TARGET_MOCAP_BODY}_site")
    return info, site_ids


def _frame_targets(fingertips_cube_i, cube_pose_i, wrist_world_i, wrist_offset):
    """Compute the world-frame target poses for this frame.

    Returns (wrist_pos, wrist_quat_wxyz, cube_pos, cube_quat_wxyz,
             {finger_name: (pos, quat_wxyz)}) or None if the frame is invalid.
    """
    if (fingertips_cube_i is None or cube_pose_i is None
            or wrist_world_i is None):
        return None

    T_world_wrist_anchor = apply_wrist_offset(wrist_world_i, wrist_offset)
    wrist_pos, wrist_quat = _T_to_mocap(T_world_wrist_anchor)

    cube_pose = np.asarray(cube_pose_i, dtype=float)
    cube_pos = cube_pose[:3].copy()
    cube_quat = cube_pose[3:7].copy()  # already wxyz from recording

    T_world_cube = pose_to_matrix(cube_pose)
    # `map_cube_frame_target` expects T_world_wrist (not the anchor), because
    # OFFLINE_TARGET_OFFSETS are wrist-frame quantities and the rotation
    # needed is the wrist rotation (anchor differs only by a translation).
    quat_wrist_xyzw = R.from_matrix(np.asarray(wrist_world_i)[:3, :3]).as_quat()

    fingertip_targets = {}
    raw = np.asarray(fingertips_cube_i, dtype=float)
    for j, ft in enumerate(FINGER_TIPS):
        pos_world, q_world = map_cube_frame_target(
            raw[j], quat_wrist_xyzw, ft.finger,
            T_world_cube, np.asarray(wrist_world_i, dtype=float),
            offsets=OFFLINE_TARGET_OFFSETS,
        )
        fingertip_targets[ft.target_body] = (
            pos_world, _xyzw_to_wxyz(q_world.as_quat()))

    return wrist_pos, wrist_quat, cube_pos, cube_quat, fingertip_targets


def _set_mocap_targets(model, data, mocap_idx, targets):
    wrist_pos, wrist_quat, cube_pos, cube_quat, ft_targets = targets
    data.mocap_pos[mocap_idx[WRIST_MOCAP_BODY]] = wrist_pos
    data.mocap_quat[mocap_idx[WRIST_MOCAP_BODY]] = wrist_quat
    data.mocap_pos[mocap_idx[CUBE_TARGET_MOCAP_BODY]] = cube_pos
    data.mocap_quat[mocap_idx[CUBE_TARGET_MOCAP_BODY]] = cube_quat
    for target_body, (pos, quat) in ft_targets.items():
        idx = mocap_idx[target_body]
        data.mocap_pos[idx] = pos
        data.mocap_quat[idx] = quat


def _pin_wrist_qpos(data, wrist_pos, wrist_quat_wxyz, *, enabled=True):
    """Hard-set the wrist freejoint qpos. Suppresses finger-pull drift."""
    if not enabled:
        return
    data.qpos[0:3] = wrist_pos
    data.qpos[3:7] = wrist_quat_wxyz
    data.qvel[0:6] = 0.0


def _site_residual(model, data, site_ids):
    """L2 residual summed over all equality-paired sites (metres + radians)."""
    total = 0.0
    pairs = [("cube_site", f"{CUBE_TARGET_MOCAP_BODY}_site")]
    pairs += [(f"{ft.finger}_site", f"{ft.target_body}_site") for ft in FINGER_TIPS]
    for a, b in pairs:
        total += float(np.linalg.norm(
            data.site_xpos[site_ids[a]] - data.site_xpos[site_ids[b]]))
    return total


def _initial_guess_search(model, data, mocap_idx, site_ids, targets, *,
                          n_guesses=NUM_INITIAL_GUESSES,
                          steps=INITIAL_GUESS_STEPS, seed=0,
                          pin_wrist=True):
    """Try N random initial qpos guesses, keep the one with smallest mocap residual.

    The wrist + cube qpos are pinned to the targets — only finger joints are
    randomised. Mirrors SPIDER's frame-0 search loop.
    """
    _set_mocap_targets(model, data, mocap_idx, targets)
    wrist_pos, wrist_quat, cube_pos, cube_quat, _ = targets

    rng = np.random.default_rng(seed)
    best_qpos = None
    best_res = np.inf
    for _ in range(n_guesses):
        data.qpos[:] = 0.0
        _pin_wrist_qpos(data, wrist_pos, wrist_quat, enabled=pin_wrist)
        # 20 finger joints sit at qpos[7:27]; cube qpos sits at qpos[27:34].
        # Sample finger joints uniformly within their joint limits.
        for jid in range(model.njnt):
            if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE,
                                        mujoco.mjtJoint.mjJNT_SLIDE):
                qa = model.jnt_qposadr[jid]
                if model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    data.qpos[qa] = rng.uniform(lo, hi)
        data.qpos[27:30] = cube_pos
        data.qpos[30:34] = cube_quat
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            _pin_wrist_qpos(data, wrist_pos, wrist_quat, enabled=pin_wrist)
        res = _site_residual(model, data, site_ids)
        if res < best_res:
            best_res = res
            best_qpos = data.qpos.copy()
    data.qpos[:] = best_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return best_res


def _step_frame(model, data, mocap_idx, targets, *,
                n_steps=STEPS_PER_FRAME, pin_wrist=True):
    _set_mocap_targets(model, data, mocap_idx, targets)
    wrist_pos, wrist_quat, _, _, _ = targets
    _pin_wrist_qpos(data, wrist_pos, wrist_quat, enabled=pin_wrist)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        _pin_wrist_qpos(data, wrist_pos, wrist_quat, enabled=pin_wrist)


def _run_session(session_dir: Path, *, show_viewer=False, force=False,
                 wrist_offset=DEFAULT_WRIST_OFFSET,
                 pin_wrist=False, loop=False,
                 output_name="retargeted_sequence.npz"):
    # --loop is interactive observation only; never overwrite the canonical file.
    write_output = not loop
    output_path = session_dir / output_name
    if write_output and output_path.exists() and not force:
        print(f"  [skip] {output_path.name} already exists (--force to overwrite)")
        return

    proc_path = session_dir / "processed_sequence_with_proxy.npz"
    if not proc_path.exists():
        print(f"  [error] {proc_path} not found")
        return
    proc = np.load(proc_path, allow_pickle=True)

    timeline = np.asarray(proc["timeline"], dtype=float)
    N = len(timeline)
    fingertips_cube = proc["right_fingertips_cube"]
    cube_pose_world = proc["cube_pose_world"]
    wrist_world = proc["right_wrist_world"]

    wrist_offset = np.asarray(wrist_offset, dtype=float)

    model, data = compile_ik(sim_dt=SIM_DT)
    mocap_idx, site_ids = _make_index(model)

    retargeted_qpos = np.full((N, 27), np.nan)
    cube_pose_used = np.full((N, 7), np.nan)

    first_i = next(
        (i for i in range(N) if _frame_targets(
            fingertips_cube[i], cube_pose_world[i],
            wrist_world[i], wrist_offset) is not None),
        None,
    )
    if first_i is None:
        print("  [error] no valid frame to warm up on")
        return

    targets0 = _frame_targets(
        fingertips_cube[first_i], cube_pose_world[first_i],
        wrist_world[first_i], wrist_offset)
    print(f"  Initial-guess search on frame {first_i} "
          f"({NUM_INITIAL_GUESSES} × {INITIAL_GUESS_STEPS} steps)…")
    best_res = _initial_guess_search(
        model, data, mocap_idx, site_ids, targets0, pin_wrist=pin_wrist)
    print(f"  best frame-0 residual: {best_res*1000:.1f} mm (summed over sites)")
    print(f"  wrist pinning: {'ON (forces qpos to target)' if pin_wrist else 'OFF (default)'}")
    print(f"  Retargeting {N} frames ({STEPS_PER_FRAME} mj_step / frame)"
          f"{', looping' if loop else ''}…")

    def _emit(i):
        retargeted_qpos[i] = data.qpos[:27].copy()
        cube_pose_used[i] = data.qpos[27:34].copy()

    def _process(i):
        t = _frame_targets(fingertips_cube[i], cube_pose_world[i],
                           wrist_world[i], wrist_offset)
        if t is None:
            return False
        _step_frame(model, data, mocap_idx, t, pin_wrist=pin_wrist)
        return True

    def _viewer_pass(viewer, rate):
        for i in range(N):
            if not viewer.is_running():
                return False
            if _process(i):
                _emit(i)
            viewer.sync()
            rate.sleep()
            if (i + 1) % 50 == 0:
                res_mm = _site_residual(model, data, site_ids) * 1000
                print(f"    [{i+1}/{N}] residual {res_mm:.1f} mm")
        return True

    if show_viewer:
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            rate = RateLimiter(frequency=PLAYBACK_HZ, warn=False)
            while True:
                if not _viewer_pass(viewer, rate):
                    break
                if not loop:
                    break
                print("    --- looping ---")
    else:
        for i in range(N):
            if _process(i):
                _emit(i)
            if (i + 1) % 50 == 0:
                res_mm = _site_residual(model, data, site_ids) * 1000
                print(f"    [{i+1}/{N}] residual {res_mm:.1f} mm")

    if write_output:
        np.savez_compressed(
            output_path,
            timeline=timeline,
            retargeted_qpos=retargeted_qpos,
            cube_pose_used=cube_pose_used,
            wrist_offset=wrist_offset,
        )
        print(f"  Saved: {output_path}")
    else:
        print(f"  [skip-save] interactive mode (loop or no-pin-wrist) — no file written")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dirs", nargs="+", type=Path)
    parser.add_argument("--viewer", action="store_true",
                        help="Show interactive viewer while retargeting.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the viewer pass (requires --viewer; no file is written).")
    parser.add_argument("--pin-wrist", action="store_true",
                        help="Force the wrist qpos to the target each step. "
                             "Default off — finger-spring reaction looks more natural.")
    args = parser.parse_args(argv)

    if args.loop and not args.viewer:
        parser.error("--loop requires --viewer")

    for session_dir in args.session_dirs:
        print(f"[{session_dir}]")
        _run_session(session_dir, show_viewer=args.viewer, force=args.force,
                     loop=args.loop, pin_wrist=args.pin_wrist)


if __name__ == "__main__":
    main()
