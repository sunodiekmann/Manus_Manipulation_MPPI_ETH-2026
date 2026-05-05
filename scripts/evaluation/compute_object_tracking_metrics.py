"""
Stage 0.5 — Object tracking metrics.

Open-loop physics rollout: drives the retargeted hand trajectory in a physics-enabled
model (cube as free body, wrist welded to tracked pose) and measures how closely the
simulated cube trajectory matches the reference.

Wrist is pinned each frame via the wrist_mocap body + equality weld in the physics model.
Finger joints are driven by position actuators (kp=5) with targets from retargeted_qpos.

Metrics (SPIDER convention, Sec 3.1):
  E_pos  — mean position error [cm]
  E_rot  — mean rotation error [deg]

Inputs:
  recordings/<session>/retargeted_sequence_cube_relative.npz  (default)
  recordings/<session>/retargeted_sequence_baseline.npz        (--baseline)
  recordings/<session>/processed_sequence_with_proxy.npz

Output:
  Console table; optionally --save writes metrics_object_tracking.json per session.

Usage:
  python -m scripts.evaluation.compute_object_tracking_metrics recordings/cube_yaw
  python -m scripts.evaluation.compute_object_tracking_metrics recordings/cube_yaw --baseline
  python -m scripts.evaluation.compute_object_tracking_metrics recordings/cube_yaw recordings/cube_pitch --save
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.utils.transforms import pose_to_matrix

PHYSICS_MODEL_PATH = "/home/suno/robohand2/assets/output_mocap_physics.xml"

# Physics substeps per control frame — model timestep 0.002 s, recording at 20 Hz
_N_SUBSTEPS = 25


def _set_wrist_mocap(model, data, T_world_wrist):
    """Drive the wrist_mocap body to the tracked wrist transform."""
    wrist_id = model.body("wrist_mocap").mocapid[0]
    data.mocap_pos[wrist_id] = T_world_wrist[:3, 3]
    quat_xyzw = R.from_matrix(T_world_wrist[:3, :3]).as_quat()
    # MuJoCo mocap_quat is wxyz
    data.mocap_quat[wrist_id] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


def _pose7_to_matrix(p7):
    """[x,y,z,qw,qx,qy,qz] → 4×4."""
    return pose_to_matrix(np.asarray(p7, dtype=float))


def _quat_angle_error(q_sim_wxyz, q_ref_wxyz):
    """Rotation angle between two wxyz quaternions [rad]."""
    q_sim = np.asarray(q_sim_wxyz, dtype=float)
    q_ref = np.asarray(q_ref_wxyz, dtype=float)
    dot = abs(np.dot(q_sim, q_ref))
    dot = np.clip(dot, 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def rollout_session(session_dir: Path, retargeted_name: str, model, data):
    """Physics rollout for one session.

    Returns (e_pos_cm_per_frame, e_rot_deg_per_frame) arrays, one value per
    valid (non-NaN) frame.
    """
    ret_path = session_dir / retargeted_name
    proc_path = session_dir / "processed_sequence_with_proxy.npz"

    if not ret_path.exists():
        print(f"  [skip] {ret_path.name} not found")
        return None, None
    if not proc_path.exists():
        print(f"  [skip] processed_sequence_with_proxy.npz not found")
        return None, None

    ret = np.load(ret_path)
    proc = np.load(proc_path, allow_pickle=True)

    retargeted_qpos = ret["retargeted_qpos"]           # (N, 27)
    cube_pose_ref = proc["cube_pose_world"]             # (N, 7) [x,y,z,qw,qx,qy,qz]

    # Wrist comes from the retargeted qpos (the IK-determined pose) rather than
    # the raw Manus tracking — accounts for the soft wrist task, so the physics
    # rollout starts where the kinematic reference actually placed the hand.
    N = len(retargeted_qpos)
    wrist_world = np.empty(N, dtype=object)
    for i in range(N):
        if np.any(np.isnan(retargeted_qpos[i])):
            wrist_world[i] = None
            continue
        T = np.eye(4)
        T[:3, 3] = retargeted_qpos[i, 0:3]
        # qpos quat order is [qw, qx, qy, qz]
        qw, qx, qy, qz = retargeted_qpos[i, 3:7]
        T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
        wrist_world[i] = T

    # Find the cube freejoint qpos address in the physics model
    cube_qposadr = int(model.joint("cube_freejoint").qposadr[0])

    # Find the first valid frame (both wrist and cube tracked, retargeted not NaN)
    first_valid = next(
        (i for i in range(N)
         if not np.any(np.isnan(retargeted_qpos[i]))
         and cube_pose_ref[i] is not None
         and wrist_world[i] is not None),
        None,
    )
    if first_valid is None:
        print("  [error] no valid frame found")
        return None, None

    # ── Reset and initialise ──────────────────────────────────────────────────
    mujoco.mj_resetData(model, data)

    T0 = np.asarray(wrist_world[first_valid], dtype=float)
    _set_wrist_mocap(model, data, T0)

    # Set hand freejoint qpos from retargeted (matches kinematic model layout)
    data.qpos[:27] = retargeted_qpos[first_valid]
    # Set cube initial pose from reference
    p_cube = np.asarray(cube_pose_ref[first_valid], dtype=float)  # [x,y,z,qw,qx,qy,qz]
    data.qpos[cube_qposadr:cube_qposadr + 7] = p_cube
    # Initialise ctrl to first-frame targets so actuators start without jerk
    data.ctrl[:] = retargeted_qpos[first_valid, 7:27]

    mujoco.mj_forward(model, data)

    # ── Rollout ───────────────────────────────────────────────────────────────
    e_pos_list = []
    e_rot_list = []

    for i in range(N):
        q = retargeted_qpos[i]
        cube_ref = cube_pose_ref[i]
        T_wrist = wrist_world[i]

        if np.any(np.isnan(q)) or cube_ref is None or T_wrist is None:
            continue  # skip frames where either tracking was lost

        T_wrist_mat = np.asarray(T_wrist, dtype=float)
        _set_wrist_mocap(model, data, T_wrist_mat)
        data.ctrl[:] = q[7:27]

        for _ in range(_N_SUBSTEPS):
            mujoco.mj_step(model, data)

        # Record simulated cube pose
        sim_pose = data.qpos[cube_qposadr:cube_qposadr + 7].copy()  # [x,y,z,qw,qx,qy,qz]
        ref_pose = np.asarray(cube_ref, dtype=float)

        pos_err_m = float(np.linalg.norm(sim_pose[:3] - ref_pose[:3]))
        rot_err_rad = _quat_angle_error(sim_pose[3:], ref_pose[3:])

        e_pos_list.append(pos_err_m * 100.0)          # → cm
        e_rot_list.append(np.degrees(rot_err_rad))    # → deg

    if not e_pos_list:
        return None, None

    return np.array(e_pos_list), np.array(e_rot_list)


def main():
    args = sys.argv[1:]
    use_baseline = "--baseline" in args
    do_save = "--save" in args
    paths = [Path(a) for a in args if not a.startswith("--")]

    if not paths:
        print(
            "Usage: python -m scripts.evaluation.compute_object_tracking_metrics <session_dir>..."
            " [--baseline] [--save]"
        )
        sys.exit(1)

    retargeted_name = (
        "retargeted_sequence_baseline.npz" if use_baseline
        else "retargeted_sequence_cube_relative.npz"
    )
    label = "baseline" if use_baseline else "cube-relative"

    model = mujoco.MjModel.from_xml_path(PHYSICS_MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"\nObject tracking metrics  [{label}]")
    print(f"{'Session':<20}  {'E_pos [cm]':>10}  {'E_rot [deg]':>11}  {'frames':>6}")
    print("-" * 54)

    all_pos, all_rot = [], []
    for p in paths:
        e_pos, e_rot = rollout_session(p, retargeted_name, model, data)
        if e_pos is None:
            print(f"  {p.name:<18}  {'—':>10}  {'—':>11}  {'—':>6}")
            continue
        mean_pos = float(np.mean(e_pos))
        mean_rot = float(np.mean(e_rot))
        all_pos.extend(e_pos.tolist())
        all_rot.extend(e_rot.tolist())
        print(f"  {p.name:<18}  {mean_pos:>10.2f}  {mean_rot:>11.2f}  {len(e_pos):>6}")

        if do_save:
            result = {
                "session": p.name,
                "retargeted_file": retargeted_name,
                "n_frames": len(e_pos),
                "E_pos_cm_mean": mean_pos,
                "E_rot_deg_mean": mean_rot,
                "E_pos_cm_per_frame": e_pos.tolist(),
                "E_rot_deg_per_frame": e_rot.tolist(),
            }
            out = p / "metrics_object_tracking.json"
            out.write_text(json.dumps(result, indent=2))
            print(f"    Saved: {out}")

    if len(paths) > 1 and all_pos:
        print("-" * 54)
        print(f"  {'MEAN':<18}  {np.mean(all_pos):>10.2f}  {np.mean(all_rot):>11.2f}  {len(all_pos):>6}")


if __name__ == "__main__":
    main()
