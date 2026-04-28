"""
Diagnostic: measure the gap between IK fingertip TARGETS and the actual robot
fingertip geom positions after retargeting, per frame and per finger.

Tells us whether the visualization "discrepancy" the user sees is:
  - small (~few mm): IK is converging fine; visual mismatch is mesh extent
  - large (>1 cm): IK is not actually hitting its targets

Usage:
    python -m scripts.calibration.diagnose_target_gap recordings/cube_yaw
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from scripts.retargeting.tesollo_constants import FINGERS, TARGET_SCALE
from scripts.retargeting.retarget_sequence_cube_relative import (
    MODEL_PATH, OFFLINE_TARGET_OFFSETS,
)
from scripts.utils.transforms import pose_to_matrix

FINGERTIP_GEOMS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.calibration.diagnose_target_gap <session_dir>")
        sys.exit(1)
    session_dir = Path(sys.argv[1])

    proc = np.load(session_dir / "processed_sequence_with_proxy.npz", allow_pickle=True)
    ret  = np.load(session_dir / "retargeted_sequence_cube_relative.npz")

    cube_pose_seq    = proc["cube_pose_world"]
    fingertips_cube  = proc["right_fingertips_cube"]
    retargeted_qpos  = ret["retargeted_qpos"]
    N = len(retargeted_qpos)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    geom_ids = [int(model.geom(g).id) for g in FINGERTIP_GEOMS]

    # Per-finger gap arrays
    gaps = np.zeros((N, 5))      # |target_world - geom_xpos_world|
    valid = np.zeros(N, dtype=bool)

    for i in range(N):
        if cube_pose_seq[i] is None or np.any(np.isnan(retargeted_qpos[i])):
            continue
        valid[i] = True
        T_cube = pose_to_matrix(np.asarray(cube_pose_seq[i], dtype=float))

        # Apply qpos and FK
        data.qpos[:len(retargeted_qpos[i])] = retargeted_qpos[i]
        mujoco.mj_forward(model, data)

        # IK target world positions (same formula as map_cube_frame_target)
        for f, finger in enumerate(FINGERS):
            pos_cube = np.asarray(fingertips_cube[i][f], dtype=float) * TARGET_SCALE \
                       + OFFLINE_TARGET_OFFSETS[finger]
            target_world = T_cube[:3, :3] @ pos_cube + T_cube[:3, 3]
            actual_world = data.geom_xpos[geom_ids[f]]
            gaps[i, f] = np.linalg.norm(actual_world - target_world)

    g = gaps[valid]
    print(f"\nSession: {session_dir.name}  ({g.shape[0]} valid frames)\n")
    print(f"  {'finger':<8}  {'mean':>8}  {'std':>8}  {'max':>8}  {'min':>8}")
    print(f"  {'':<8}  {'(mm)':>8}  {'(mm)':>8}  {'(mm)':>8}  {'(mm)':>8}")
    for f, name in enumerate(FINGERS):
        col = g[:, f] * 1000
        print(f"  {name:<8}  {col.mean():>8.2f}  {col.std():>8.2f}  {col.max():>8.2f}  {col.min():>8.2f}")
    print(f"  {'ALL':<8}  {(g*1000).mean():>8.2f}  {(g*1000).std():>8.2f}  {(g*1000).max():>8.2f}  {(g*1000).min():>8.2f}")
    print(f"\n  RMSE = {np.sqrt(((g*1000)**2).mean()):.2f} mm")
    print("\n  >> If this number is small (<5mm), IK is converging and visual mismatch")
    print("     is just fingertip-mesh extent.  If large (>1cm), IK can't hit targets.")


if __name__ == "__main__":
    main()
