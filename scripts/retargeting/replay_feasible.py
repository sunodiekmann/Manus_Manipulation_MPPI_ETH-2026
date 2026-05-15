"""Replay a Stage 2c (feasibility-IK) sequence in the MuJoCo viewer.

Loads `retargeted_sequence_feasible.npz` and plays back the IK qpos against
the same physics scene used by Stage 3 MPPI (`assets/output_mocap_physics.xml`).
Optionally overlays:
  - the kinematic Stage 2a reference cube (ghost orange) so you can see how
    much Pass 1 smoothing nudged the cube
  - per-frame override markers (the frames where lift targets were inserted)

Usage:
    python -m scripts.retargeting.replay_feasible recordings/<session>
    python -m scripts.retargeting.replay_feasible recordings/<session> --speed 0.5
    python -m scripts.retargeting.replay_feasible recordings/<session> --no-ref
    python -m scripts.retargeting.replay_feasible recordings/<session> --compare
        # side-by-side against retargeted_sequence_cube_relative.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"


def _apply_state(model, data, qpos_hand, cube_pose):
    """Set hand qpos + cube qpos and sync wrist mocap to the freejoint."""
    data.qpos[:27] = qpos_hand
    data.qpos[27:34] = cube_pose
    data.qvel[:] = 0.0
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_mocap")
    wrist_mid = int(model.body_mocapid[wrist_bid])
    data.mocap_pos[wrist_mid] = qpos_hand[0:3]
    data.mocap_quat[wrist_mid] = qpos_hand[3:7]
    mujoco.mj_forward(model, data)


def _draw_ghost_cube(scene, cube_pose, half_size=0.035,
                     rgba=(1.0, 0.7, 0.0, 0.35)):
    """Append a translucent cube to the user scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    rgba_arr = np.array(rgba, dtype=np.float32)
    size = np.array([half_size, half_size, half_size], dtype=np.float64)
    pos = np.asarray(cube_pose[0:3], dtype=np.float64)
    R = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R, np.asarray(cube_pose[3:7], dtype=np.float64))
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=size, pos=pos, mat=R, rgba=rgba_arr)
    g.label = b""
    scene.ngeom += 1


def _draw_override_marker(scene, qpos_hand, pos_offset=(0.0, 0.0, 0.18)):
    """Append a red sphere above the wrist to flag override frames."""
    if scene.ngeom >= scene.maxgeom:
        return
    rgba = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)
    size = np.array([0.012, 0.012, 0.012], dtype=np.float64)
    pos = np.asarray(qpos_hand[0:3], dtype=np.float64) + np.asarray(pos_offset)
    R = np.eye(3).reshape(-1).astype(np.float64)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=size, pos=pos, mat=R, rgba=rgba)
    g.label = b""
    scene.ngeom += 1


def replay(session_dir: Path, speed: float, show_ref: bool, loop: bool,
           compare: bool):
    feas_path = session_dir / "retargeted_sequence_feasible.npz"
    if not feas_path.exists():
        raise FileNotFoundError(f"missing {feas_path}")
    feas = np.load(feas_path)
    timeline = np.asarray(feas["timeline"], dtype=float)
    qpos = np.asarray(feas["retargeted_qpos"], dtype=float)
    cube_smoothed = np.asarray(feas["cube_pose_smoothed"], dtype=float)
    snap_weight = (
        np.asarray(feas["snap_weight"], dtype=float)
        if "snap_weight" in feas.files else np.zeros((qpos.shape[0], 5))
    )
    N = qpos.shape[0]

    cube_ref = None
    if show_ref:
        ref_path = session_dir / "retargeted_sequence_cube_relative.npz"
        if ref_path.exists():
            cube_ref = np.asarray(np.load(ref_path)["cube_pose_used"], dtype=float)
        else:
            print(f"  [warn] no Stage 2a reference at {ref_path.name}; ghost disabled")

    compare_qpos = None
    compare_cube = None
    if compare:
        ref_path = session_dir / "retargeted_sequence_cube_relative.npz"
        if ref_path.exists():
            rel = np.load(ref_path)
            compare_qpos = np.asarray(rel["retargeted_qpos"], dtype=float)
            compare_cube = np.asarray(rel["cube_pose_used"], dtype=float)
        else:
            print(f"  [warn] no Stage 2a output for --compare")
            compare = False

    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    data = mujoco.MjData(model)

    dt_per_frame = float(np.diff(timeline).mean()) if N > 1 else 0.05
    sleep_dt = dt_per_frame / max(speed, 1e-6)

    n_snap_frames = int((snap_weight > 0).any(axis=1).sum())
    print(f"  replaying {N} frames  dt={dt_per_frame*1000:.1f} ms  ×{speed}"
          f"  ghost-ref={show_ref and cube_ref is not None}"
          f"  snap-active={n_snap_frames} frames  loop={loop}")
    print(f"  session: {session_dir.name}")
    if compare:
        print("  press TAB during playback to toggle (mujoco viewer hotkey, "
              "not implemented here). Re-run without --compare to swap.")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            for t in range(N):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()

                q_t = qpos[t]
                cp_t = cube_smoothed[t]
                if compare and compare_qpos is not None:
                    # Show Stage 2a in this frame instead (alternating could be
                    # a future enhancement; for now show only one).
                    q_t = compare_qpos[t]
                    cp_t = compare_cube[t]

                if np.any(np.isnan(q_t)) or np.any(np.isnan(cp_t)):
                    continue
                _apply_state(model, data, q_t, cp_t)

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if cube_ref is not None and not np.any(np.isnan(cube_ref[t])):
                        _draw_ghost_cube(viewer.user_scn, cube_ref[t])
                    if (snap_weight[t] > 0.5).any():
                        _draw_override_marker(viewer.user_scn, q_t)

                mujoco.mj_camlight(model, data)
                viewer.sync()
                elapsed = time.perf_counter() - t0
                remaining = sleep_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            if not loop:
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_dir", type=Path)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-ref", dest="show_ref", action="store_false",
                   help="hide the ghost Stage 2a (recorded) cube")
    p.add_argument("--once", dest="loop", action="store_false",
                   help="play once instead of looping")
    p.add_argument("--compare", action="store_true",
                   help="show Stage 2a qpos+cube instead of Stage 2c "
                        "(quick side-by-side via two terminals)")
    args = p.parse_args()
    replay(args.session_dir, speed=args.speed, show_ref=args.show_ref,
           loop=args.loop, compare=args.compare)


if __name__ == "__main__":
    main()
