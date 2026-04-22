"""
Extend a processed sequence with derived fingertip geometry (proxy geometry).

Input:  recordings/<session>/processed_sequence.npz
Output: recordings/<session>/processed_sequence_with_proxy.npz

What it does:
  - Extracts the 5 fingertip positions (thumb → little) from the 25-joint hand
    skeleton for every frame, in both wrist frame and world frame.
  - Computes the fingertip direction vectors (unit vectors from the pre-tip joint
    to the tip joint), also in both frames.
  - Transforms fingertip positions and wrist transforms into the cube frame using
    the live cube pose, producing cube-relative representations that are useful
    for calibration evaluation and imitation learning.
  - All outputs use object arrays (None for frames where a stream was unavailable).

Usage:
    python -m scripts.processing.compute_proxy_geometry                                             # uses hardcoded path
    python -m scripts.processing.compute_proxy_geometry recordings/<session>/processed_sequence.npz
"""
import numpy as np
from pathlib import Path

from scripts.utils.transforms import transform_points, safe_inverse_transform, pose_to_matrix

INPUT_PATH = Path("recordings/statics/session_20260421_150421/processed_sequence.npz")
OUTPUT_PATH = Path("recordings/statics/session_20260421_150421/processed_sequence_with_proxy.npz")

import sys as _sys
if len(_sys.argv) > 1:
    INPUT_PATH = Path(_sys.argv[1])
    OUTPUT_PATH = INPUT_PATH.parent / "processed_sequence_with_proxy.npz"

TIP_IDXS = {
    "thumb": 4,
    "index": 9,
    "middle": 14,
    "ring": 19,
    "little": 24,
}

PRE_TIP_IDXS = {
    "thumb": 3,
    "index": 8,
    "middle": 13,
    "ring": 18,
    "little": 23,
}

def extract_fingertips(joints):
    if joints is None:
        return None
    return np.stack([joints[idx] for idx in TIP_IDXS.values()], axis=0)


def extract_fingertip_dirs(joints):
    if joints is None:
        return None

    dirs = []
    for finger in TIP_IDXS.keys():
        p0 = joints[PRE_TIP_IDXS[finger]]
        p1 = joints[TIP_IDXS[finger]]
        d = p1 - p0
        n = np.linalg.norm(d)
        if n < 1e-8:
            dirs.append(np.zeros(3))
        else:
            dirs.append(d / n)
    return np.stack(dirs, axis=0)


def transform_dirs(R, dirs):
    if dirs is None:
        return None
    return (R @ dirs.T).T


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)

    timeline = data["timeline"]

    left_joints_wrist = data["left_joints_wrist"]
    right_joints_wrist = data["right_joints_wrist"]

    left_joints_world = data["left_joints_world"]
    right_joints_world = data["right_joints_world"]

    left_wrist_world = data["left_wrist_world"]
    right_wrist_world = data["right_wrist_world"]

    cube_pose_world = data["cube_pose_world"]

    left_fingertips_wrist = []
    right_fingertips_wrist = []

    left_fingertips_world = []
    right_fingertips_world = []

    left_fingerdirs_wrist = []
    right_fingerdirs_wrist = []

    left_fingerdirs_world = []
    right_fingerdirs_world = []

    left_fingertips_cube = []
    right_fingertips_cube = []

    left_wrist_cube = []
    right_wrist_cube = []

    for i in range(len(timeline)):
        ljw = left_joints_wrist[i]
        rjw = right_joints_wrist[i]
        ljworld = left_joints_world[i]
        rjworld = right_joints_world[i]

        T_world_left_wrist = left_wrist_world[i]
        T_world_right_wrist = right_wrist_world[i]

        cube_pose = cube_pose_world[i]

        l_tip_wrist = extract_fingertips(ljw)
        r_tip_wrist = extract_fingertips(rjw)

        l_tip_world = extract_fingertips(ljworld)
        r_tip_world = extract_fingertips(rjworld)

        l_dir_wrist = extract_fingertip_dirs(ljw)
        r_dir_wrist = extract_fingertip_dirs(rjw)

        l_dir_world = extract_fingertip_dirs(ljworld)
        r_dir_world = extract_fingertip_dirs(rjworld)

        T_world_cube = None
        if cube_pose is not None:
            T_world_cube = pose_to_matrix(cube_pose)

        T_cube_world = safe_inverse_transform(T_world_cube)

        if T_cube_world is not None and l_tip_world is not None:
            l_tip_cube = transform_points(T_cube_world, l_tip_world)
        else:
            l_tip_cube = None

        if T_cube_world is not None and r_tip_world is not None:
            r_tip_cube = transform_points(T_cube_world, r_tip_world)
        else:
            r_tip_cube = None

        if T_cube_world is not None and T_world_left_wrist is not None:
            T_cube_left_wrist = T_cube_world @ T_world_left_wrist
        else:
            T_cube_left_wrist = None

        if T_cube_world is not None and T_world_right_wrist is not None:
            T_cube_right_wrist = T_cube_world @ T_world_right_wrist
        else:
            T_cube_right_wrist = None

        left_fingertips_wrist.append(l_tip_wrist)
        right_fingertips_wrist.append(r_tip_wrist)

        left_fingertips_world.append(l_tip_world)
        right_fingertips_world.append(r_tip_world)

        left_fingerdirs_wrist.append(l_dir_wrist)
        right_fingerdirs_wrist.append(r_dir_wrist)

        left_fingerdirs_world.append(l_dir_world)
        right_fingerdirs_world.append(r_dir_world)

        left_fingertips_cube.append(l_tip_cube)
        right_fingertips_cube.append(r_tip_cube)

        left_wrist_cube.append(T_cube_left_wrist)
        right_wrist_cube.append(T_cube_right_wrist)

    output = dict(data.items())
    output.update({
        "left_fingertips_wrist": np.array(left_fingertips_wrist, dtype=object),
        "right_fingertips_wrist": np.array(right_fingertips_wrist, dtype=object),
        "left_fingertips_world": np.array(left_fingertips_world, dtype=object),
        "right_fingertips_world": np.array(right_fingertips_world, dtype=object),
        "left_fingerdirs_wrist": np.array(left_fingerdirs_wrist, dtype=object),
        "right_fingerdirs_wrist": np.array(right_fingerdirs_wrist, dtype=object),
        "left_fingerdirs_world": np.array(left_fingerdirs_world, dtype=object),
        "right_fingerdirs_world": np.array(right_fingerdirs_world, dtype=object),
        "left_fingertips_cube": np.array(left_fingertips_cube, dtype=object),
        "right_fingertips_cube": np.array(right_fingertips_cube, dtype=object),
        "left_wrist_cube": np.array(left_wrist_cube, dtype=object),
        "right_wrist_cube": np.array(right_wrist_cube, dtype=object),
    })

    np.savez_compressed(OUTPUT_PATH, **output)
    print(f"Saved proxy geometry dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()