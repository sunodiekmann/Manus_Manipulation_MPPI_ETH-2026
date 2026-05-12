"""Replay a dynamic-retargeted sequence in the MuJoCo viewer.

Plays back the qpos + cube trajectory produced by
`scripts.retargeting.optimize_sequence_dynamic` against the same physics scene
the MPPI ran in (`assets/output_mocap_physics.xml`).

Optionally also shows the kinematic reference cube as a translucent ghost so
you can see how far the sim cube has drifted from the IK plan.

Usage:
    python -m scripts.retargeting.replay_dynamic recordings/<session>
    python -m scripts.retargeting.replay_dynamic recordings/<session> --speed 0.5
    python -m scripts.retargeting.replay_dynamic recordings/<session> --no-ref
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
    """Set hand qpos + cube qpos and keep the wrist mocap in sync with the
    hand freejoint so the weld equality stays satisfied during mj_forward."""
    data.qpos[:27] = qpos_hand
    data.qpos[27:34] = cube_pose
    data.qvel[:] = 0.0
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_mocap")
    wrist_mid = int(model.body_mocapid[wrist_bid])
    data.mocap_pos[wrist_mid] = qpos_hand[0:3]
    data.mocap_quat[wrist_mid] = qpos_hand[3:7]
    mujoco.mj_forward(model, data)


def _draw_ghost_cube(scene, cube_pose_ref, half_size=0.035):
    """Append a wireframe box at the reference cube pose to the user scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    rgba = np.array([1.0, 0.7, 0.0, 0.6], dtype=np.float32)
    size = np.array([half_size, half_size, half_size], dtype=np.float64)
    pos = np.asarray(cube_pose_ref[0:3], dtype=np.float64)
    R = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R, np.asarray(cube_pose_ref[3:7], dtype=np.float64))
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=size, pos=pos, mat=R, rgba=rgba)
    # wireframe-ish: turn off shading via a high transparency
    g.label = b""
    scene.ngeom += 1


def replay(session_dir: Path, speed: float, show_ref: bool, loop: bool):
    dyn_path = session_dir / "retargeted_sequence_dynamic.npz"
    if not dyn_path.exists():
        raise FileNotFoundError(f"missing {dyn_path}")
    dyn = np.load(dyn_path)
    timeline = np.asarray(dyn["timeline"], dtype=float)
    qpos_sim = np.asarray(dyn["retargeted_qpos"], dtype=float)
    cube_sim = np.asarray(dyn["cube_traj"], dtype=float)
    N = qpos_sim.shape[0]

    cube_ref = None
    if show_ref:
        ref_path = session_dir / "retargeted_sequence_cube_relative.npz"
        if ref_path.exists():
            cube_ref = np.asarray(np.load(ref_path)["cube_pose_used"], dtype=float)
        else:
            print(f"  [warn] no kinematic reference at {ref_path.name}; ghost disabled")

    rew_max = dyn.get("rew_max")
    rew_mean = dyn.get("rew_mean")

    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    data = mujoco.MjData(model)

    dt_per_frame = float(np.diff(timeline).mean()) if N > 1 else 0.05
    sleep_dt = dt_per_frame / max(speed, 1e-6)

    print(f"  replaying {N} frames  dt={dt_per_frame*1000:.1f} ms  ×{speed}"
          f"  ghost-ref={show_ref and cube_ref is not None}  loop={loop}")
    print(f"  session: {session_dir.name}")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            for t in range(N):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                _apply_state(model, data, qpos_sim[t], cube_sim[t])

                if cube_ref is not None:
                    with viewer.lock():
                        # reset user-scene geoms each frame
                        viewer.user_scn.ngeom = 0
                        _draw_ghost_cube(viewer.user_scn, cube_ref[t])

                if rew_max is not None:
                    pass  # could show on-screen if desired

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
                   help="hide the ghost reference cube")
    p.add_argument("--once", dest="loop", action="store_false",
                   help="play through once instead of looping")
    args = p.parse_args()
    replay(args.session_dir, speed=args.speed, show_ref=args.show_ref, loop=args.loop)


if __name__ == "__main__":
    main()
