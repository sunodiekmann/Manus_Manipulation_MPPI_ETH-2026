import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser(description="Plot wrist and hand pose data")
parser.add_argument(
    "folder",
    type=str,
    help="Path to folder containing .npy files"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="Save figures instead of only showing them"
)

args = parser.parse_args()
folder = args.folder
save_figs = args.save

out_dir = os.path.join(folder, "plots")
if save_figs:
    os.makedirs(out_dir, exist_ok=True)

def maybe_save(fig, name):
    if save_figs:
        fig.savefig(os.path.join(out_dir, name), dpi=200)

def transform_pose(pose_array):
    """
    Transforms a [N, 7] array (x, y, z, qx, qy, qz, qw) 
    transform by one 90deg rotation around its own y axis, 
    then one 90deg rotation around its own x axis 

    """
    # 1. Define the local rotations
    # Rotation about Y by 90 degrees
    r_y = R.from_euler('y', -90, degrees=True)
    # Rotation about X by 90 degrees
    r_x = R.from_euler('x', -90, degrees=True)
    r_z = R.from_euler('z', 60, degrees=True)
    
    # Combine them: Y first, then X (intrinsic/local)
    # For local rotations, we multiply in order of application: q_total = q_y * q_x
    q_rot = r_y * r_x
    # q_rot = r_z
    
    # Extract positions and orientations
    pos = pose_array[:, :3]
    ori = R.from_quat(pose_array[:, 3:], scalar_first=False)
    
    # 2. Rotate the orientations
    # Assuming the rotation is applied in the global/parent frame
    transformed_ori = R.from_euler('z', -90, degrees=True) * ori * r_z #* q_rot
    
    # Reconstruct the [N, 7] array
    transformed_pose = np.zeros_like(pose_array)
    transformed_pose[:, :3] = pos
    transformed_pose[:, 3:] = transformed_ori.as_quat(scalar_first=False)
    
    return transformed_pose

def plot_wrist():
    timestamps = np.load(os.path.join(folder, "times.npy"))
    
    actual_wrist = np.load(os.path.join(folder, "actual_wrist_poses.npy"))
    actual_wrist = transform_pose(actual_wrist)

    desired_wrist = np.load(os.path.join(folder, "desired_wrist_poses.npy"))
    # desired_wrist = transform_pose(desired_wrist)

    t = timestamps

    # =========================
    # WRIST POSITION (XYZ)
    # =========================
    labels = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axs[i].plot(t, actual_wrist[:, i], label="actual")
        axs[i].plot(t, desired_wrist[:, i], "--", label="desired")
        axs[i].set_ylabel(labels[i])
        axs[i].grid()

    axs[0].legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Wrist Position")
    fig.tight_layout()
    maybe_save(fig, "wrist_position.png")

    # =========================
    # WRIST POSITION ERROR
    # =========================
    wrist_pos_error = np.linalg.norm(
        actual_wrist[:, :3] - desired_wrist[:, :3],
        axis=1
    )

    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, wrist_pos_error)
    plt.xlabel("Time step")
    plt.ylabel("Position error")
    plt.title("Wrist Position Error")
    plt.grid()
    maybe_save(fig, "wrist_position_error.png")

    # =========================
    # WRIST ORIENTATION (EULER)
    # =========================
    actual_euler = R.from_quat(actual_wrist[:, 3:7], scalar_first=False).as_euler("xyz", degrees=True)
    desired_euler = R.from_quat(desired_wrist[:, 3:7], scalar_first=True).as_euler("xyz", degrees=True)

    labels = ["roll (deg)", "pitch (deg)", "yaw (deg)"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axs[i].plot(t, actual_euler[:, i], label="actual")
        axs[i].plot(t, desired_euler[:, i], "--", label="desired")
        axs[i].set_ylabel(labels[i])
        axs[i].grid()

    axs[0].legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Wrist Orientation (Euler)")
    fig.tight_layout()
    maybe_save(fig, "wrist_orientation_euler.png")

    # =========================
    # WRIST ORIENTATION ERROR
    # =========================
    rel_rot = (
        R.from_quat(desired_wrist[:, 3:7], scalar_first=True).inv()
        * R.from_quat(actual_wrist[:, 3:7], scalar_first=False)
    )
    angle_error = rel_rot.magnitude() * 180 / np.pi

    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, angle_error)
    plt.xlabel("Time step")
    plt.ylabel("Angular error (deg)")
    plt.title("Wrist Orientation Error")
    plt.grid()
    maybe_save(fig, "wrist_orientation_error.png")


def plot_hand():
    timestamps = np.load(os.path.join(folder, "times.npy"))

    actual_hand = np.load(os.path.join(folder, "actual_hand_poses.npy"))
    desired_hand = np.load(os.path.join(folder, "desired_hand_poses.npy"))
    
    t = timestamps

    # =========================
    # HAND JOINTS (20D)
    # =========================
    fig, axs = plt.subplots(5, 4, figsize=(16, 10), sharex=True)
    axs = axs.flatten()

    for i in range(20):
        axs[i].plot(t, actual_hand[:, i], label="actual")
        axs[i].plot(t, desired_hand[:, i], "--", label="desired")
        axs[i].set_title(f"Joint {i}")
        axs[i].grid()

    axs[0].legend()
    fig.suptitle("Hand Joint Positions")
    fig.tight_layout()
    maybe_save(fig, "hand_joints.png")

    # =========================
    # HAND ERROR NORM
    # =========================
    hand_error = actual_hand - desired_hand
    hand_error_norm = np.linalg.norm(hand_error, axis=1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, hand_error_norm)
    plt.xlabel("Time step")
    plt.ylabel("L2 error")
    plt.title("Hand Pose Tracking Error")
    plt.grid()
    maybe_save(fig, "hand_error_norm.png")

# plot_hand()
plot_wrist()

plt.show()
