import numpy as np
import mujoco
import mink
from scipy.spatial.transform import Rotation as R


def set_hand_base_pose(model, data, T_world_from_wrist):
    pos = T_world_from_wrist[:3, 3]
    quat_xyzw = R.from_matrix(T_world_from_wrist[:3, :3]).as_quat()

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    data.qpos[0:3] = pos
    data.qpos[3:7] = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=float,
    )
    mujoco.mj_forward(model, data)


def set_mocap_body_pose(model, data, body_name: str, position, quat_xyzw):
    mocap_id = model.body(body_name).mocapid[0]
    data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)
    data.mocap_quat[mocap_id] = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=float,
    )


def set_mocap_body_pose_from_matrix(model, data, body_name: str, T_world_from_body):
    pos = T_world_from_body[:3, 3]
    quat_xyzw = R.from_matrix(T_world_from_body[:3, :3]).as_quat()
    set_mocap_body_pose(model, data, body_name, pos, quat_xyzw)


def initialize_fingertip_target_mocaps(model, data, fingers):
    for finger in fingers:
        mujoco.mj_forward(model, data)
        mink.move_mocap_to_frame(model, data, f"{finger}_target", finger, "geom")