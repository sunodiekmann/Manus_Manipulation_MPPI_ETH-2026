"""Replay a retargeted sequence in the MuJoCo viewer.

Two modes:

  Default (Stage 3 dynamic): plays back the per-tick qpos + cube trajectory
  produced by `scripts.retargeting.optimize_sequence_dynamic`. The orange
  ghost is the MPC reference (cube_ref_used) at the SAME tick — so visual
  error matches the saved pos_err_history / rot_err_history exactly.

  --stage2: plays the Stage 2 kinematic IK output directly from
  `retargeted_sequence.npz` (hand qpos + cube pose). With --cube-shift X Y Z
  also draws a green ghost at the shifted cube position so you can sanity-
  check the shift before running Stage 3.

Usage:
    python -m scripts.visualization.inspect_retargeting recordings/<session>
    python -m scripts.visualization.inspect_retargeting recordings/<session> --speed 0.5
    python -m scripts.visualization.inspect_retargeting recordings/<session> --no-ref
    python -m scripts.visualization.inspect_retargeting recordings/<session> --stage2
    python -m scripts.visualization.inspect_retargeting recordings/<session> --stage2 \\
        --cube-shift 0.02 0 -0.01
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from scripts.retargeting.kinematic_retargeting import shift_cube_in_wrist_frame

PHYSICS_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"

GHOST_REF_RGBA = (1.0, 0.7, 0.0, 0.2)   # orange — MPC reference
GHOST_SHIFTED_RGBA = (0.2, 1.0, 0.3, 0.7)  # green — shifted cube preview
CUBE_GEOM_RGBA = (0.0, 0.85, 0.9, 0.95)    # vivid cyan, nearly opaque


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


def _draw_ghost_cube(scene, cube_pose, rgba, half_size=0.035):
    """Append a translucent box at `cube_pose` (xyz + wxyz) to the user scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    size = np.array([half_size, half_size, half_size], dtype=np.float64)
    pos = np.asarray(cube_pose[0:3], dtype=np.float64)
    R = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(R, np.asarray(cube_pose[3:7], dtype=np.float64))
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=size, pos=pos, mat=R,
                        rgba=np.asarray(rgba, dtype=np.float32))
    g.label = b""
    scene.ngeom += 1


def _load_stage2(session_dir: Path, cube_shift: np.ndarray | None):
    """Returns (qpos, cube_play, cube_shifted_or_none, timeline)."""
    ref_path = session_dir / "retargeted_sequence.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"missing {ref_path}")
    rt = np.load(ref_path)
    qpos = np.asarray(rt["retargeted_qpos"], dtype=float)
    cube_play = np.asarray(rt["cube_pose_used"], dtype=float)
    timeline = np.asarray(rt["timeline"], dtype=float)
    cube_shifted = None
    if cube_shift is not None and not np.allclose(cube_shift, 0.0):
        proc_path = session_dir / "processed_sequence_with_proxy.npz"
        if not proc_path.exists():
            raise FileNotFoundError(
                f"{proc_path.name} required for --cube-shift preview")
        proc = np.load(proc_path, allow_pickle=True)
        cube_shifted = shift_cube_in_wrist_frame(
            cube_play, proc["right_wrist_world"], cube_shift)
    return qpos, cube_play, cube_shifted, timeline


def _load_stage3(session_dir: Path, tag: str | None = None):
    """Returns (qpos, cube_sim, cube_ref_or_none, timeline) — all per-tick."""
    dyn_name = (f"retargeted_sequence_dynamic__{tag}.npz" if tag
                else "retargeted_sequence_dynamic.npz")
    dyn_path = session_dir / dyn_name
    if not dyn_path.exists():
        raise FileNotFoundError(f"missing {dyn_path}")
    dyn = np.load(dyn_path)
    qpos = np.asarray(dyn["retargeted_qpos"], dtype=float)
    cube_sim = np.asarray(dyn["cube_traj"], dtype=float)
    timeline = np.asarray(dyn["timeline"], dtype=float)
    cube_ref = (np.asarray(dyn["cube_ref_used"], dtype=float)
                if "cube_ref_used" in dyn.files else None)
    if cube_ref is not None and cube_ref.shape[0] != qpos.shape[0]:
        print(f"  [warn] cube_ref_used has {cube_ref.shape[0]} frames vs "
              f"qpos {qpos.shape[0]}; ghost disabled.")
        cube_ref = None
    cfg = {}
    if "run_config" in dyn.files:
        import json
        try:
            cfg = json.loads(str(dyn["run_config"]))
        except Exception:
            cfg = {}
    cube_edge_m = cfg.get("cube_edge_m")
    cube_mass_kg = cfg.get("cube_mass_kg")
    return qpos, cube_sim, cube_ref, timeline, cube_edge_m, cube_mass_kg


def replay(session_dir: Path, *, speed: float, show_ref: bool, loop: bool,
           stage2: bool, cube_shift: np.ndarray | None, tag: str | None = None):
    cube_edge_m = cube_mass_kg = None
    if stage2:
        qpos, cube_play, cube_shifted, timeline = _load_stage2(session_dir, cube_shift)
        cube_ref_ghost = None
        mode_label = f"stage2  shift={cube_shift if cube_shift is not None else 'none'}"
    else:
        qpos, cube_play, cube_ref_ghost, timeline, cube_edge_m, cube_mass_kg = \
            _load_stage3(session_dir, tag)
        if not show_ref:
            cube_ref_ghost = None
        cube_shifted = None
        mode_label = f"stage3 (dynamic)  tag={tag if tag else 'canonical'}"

    N = qpos.shape[0]
    model = mujoco.MjModel.from_xml_path(str(PHYSICS_XML))
    if cube_edge_m is not None or cube_mass_kg is not None:
        from scripts.retargeting.build_ik_scene import apply_cube_geometry
        apply_cube_geometry(model, edge_m=cube_edge_m, mass_kg=cube_mass_kg)
        print(f"  cube override: "
              f"edge={cube_edge_m if cube_edge_m is not None else 'XML'} m  "
              f"mass={cube_mass_kg if cube_mass_kg is not None else 'XML'} kg")
    cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tracked_cube_geom")
    if cube_gid >= 0:
        model.geom_rgba[cube_gid] = CUBE_GEOM_RGBA
    for tip_body in ("thumb_tip_target", "index_tip_target", "middle_tip_target",
                     "ring_tip_target", "little_tip_target"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_body)
        if bid >= 0:
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    model.geom_rgba[gid, 3] = 0.0
    data = mujoco.MjData(model)

    dt_per_frame = float(np.diff(timeline).mean()) if N > 1 else 0.05
    sleep_dt = dt_per_frame / max(speed, 1e-6)

    print(f"  replaying {N} frames  dt={dt_per_frame*1000:.1f} ms  ×{speed}"
          f"  loop={loop}")
    print(f"  session: {session_dir.name}  mode: {mode_label}")
    if cube_shifted is not None:
        print(f"  green ghost = shifted cube target  orange ghost = (n/a in stage2)")
    elif cube_ref_ghost is not None:
        print(f"  orange ghost = MPC reference cube")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while viewer.is_running():
            for t in range(N):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                _apply_state(model, data, qpos[t], cube_play[t])

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if cube_ref_ghost is not None:
                        _draw_ghost_cube(viewer.user_scn, cube_ref_ghost[t], GHOST_REF_RGBA,
                                         half_size=(cube_edge_m or 0.07) / 2.0)
                    if cube_shifted is not None:
                        _draw_ghost_cube(viewer.user_scn, cube_shifted[t], GHOST_SHIFTED_RGBA,
                                         half_size=(cube_edge_m or 0.07) / 2.0)

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
                   help="hide the orange ghost reference cube (stage3 mode)")
    p.add_argument("--once", dest="loop", action="store_false",
                   help="play through once instead of looping")
    p.add_argument("--stage2", action="store_true",
                   help="replay the Stage 2 kinematic IK output "
                        "(retargeted_sequence.npz) instead of the dynamic npz")
    p.add_argument("--tag", type=str, default=None,
                   help="replay a tagged sweep run "
                        "(retargeted_sequence_dynamic__<tag>.npz). "
                        "Ignored with --stage2.")
    p.add_argument("--cube-shift", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="wrist-frame shift (m) to preview as a green ghost "
                        "over the original cube. Same semantics as the "
                        "--cube-shift flag of optimize_sequence_dynamic. "
                        "Only meaningful with --stage2.")
    args = p.parse_args()
    shift = (np.asarray(args.cube_shift, dtype=float)
             if args.cube_shift is not None else None)
    if shift is not None and not args.stage2:
        print("  [note] --cube-shift ignored without --stage2")
        shift = None
    replay(args.session_dir, speed=args.speed, show_ref=args.show_ref,
           loop=args.loop, stage2=args.stage2, cube_shift=shift, tag=args.tag)


if __name__ == "__main__":
    main()
