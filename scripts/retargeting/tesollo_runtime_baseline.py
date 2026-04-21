import numpy as np
import mink

from scripts.retargeting.tesollo_target_mapping import (
    map_wrist_frame_target,
    map_world_frame_target,
)
from scripts.retargeting.tesollo_mujoco_utils import set_mocap_body_pose


def initialize_targets_from_configuration(task_bundle: object):
    task_bundle.posture_task.set_target_from_configuration(task_bundle.configuration)


def update_fingertip_targets_wrist_frame(
    model,
    data,
    task_bundle,
    fingertip_positions,
    fingertip_quats,
    offsets=None,
    scale=1.0,
):
    if fingertip_positions is None or fingertip_quats is None:
        return False

    for i, finger in enumerate(task_bundle.fingers):
        pos_target, q_target = map_wrist_frame_target(
            fingertip_positions[i],
            fingertip_quats[i],
            finger,
            offsets=offsets,
            scale=scale,
        )
        set_mocap_body_pose(
            model,
            data,
            f"{finger}_target",
            pos_target,
            q_target.as_quat(),
        )

    return True


def update_fingertip_targets_world_frame(
    model,
    data,
    task_bundle,
    fingertip_positions,
    fingertip_quats,
    T_world_from_wrist,
    offsets=None,
    scale=1.0,
):
    if fingertip_positions is None or fingertip_quats is None or T_world_from_wrist is None:
        return False

    for i, finger in enumerate(task_bundle.fingers):
        pos_target, q_target = map_world_frame_target(
            fingertip_positions[i],
            fingertip_quats[i],
            finger,
            T_world_from_wrist,
            offsets=offsets,
            scale=scale,
        )
        set_mocap_body_pose(
            model,
            data,
            f"{finger}_target",
            pos_target,
            q_target.as_quat(),
        )

    return True


def update_relative_finger_targets(task_bundle):
    configuration = task_bundle.configuration

    for (idx1, idx2), task in zip(task_bundle.finger_pairs, task_bundle.relative_finger_tasks):
        finger_1 = task_bundle.fingers[idx1]
        finger_2 = task_bundle.fingers[idx2]

        T_relative = configuration.get_transform(
            f"{finger_1}_target",
            "body",
            f"{finger_2}_target",
            "body",
        )
        task.set_target(T_relative)


def reset_relative_finger_targets(task_bundle):
    for task in task_bundle.relative_finger_tasks:
        task.set_target(mink.SE3.identity())


def update_posture_target_from_joint_angles(task_bundle, angles, qpos_start_index=7):
    if angles is None:
        return

    # Build full qpos: wrist fixed at origin (identity free joint) + finger angles
    nq = task_bundle.configuration.model.nq
    target_q = np.zeros(nq)
    target_q[3] = 1.0  # quaternion w=1 for identity orientation
    target_q[qpos_start_index:qpos_start_index + len(angles)] = angles
    task_bundle.posture_task.set_target(target_q)


def update_posture_target_from_full_qpos(task_bundle, data, angles, qpos_start_index=7):
    if angles is None:
        return

    target_q = data.qpos.copy()
    target_q[qpos_start_index:qpos_start_index + len(angles)] = angles
    task_bundle.posture_task.set_target(target_q)


def update_finger_frame_tasks_from_mocaps(model, data, task_bundle):
    for finger, task in zip(task_bundle.fingers, task_bundle.finger_tasks):
        task.set_target(
            mink.SE3.from_mocap_name(model, data, f"{finger}_target")
        )


def solve_ik_step(task_bundle, dt: float, damping: float = 1e-5):
    vel = mink.solve_ik(
        task_bundle.configuration,
        task_bundle.tasks,
        dt,
        task_bundle.solver,
        damping,
        limits=task_bundle.limits,
    )
    task_bundle.configuration.integrate_inplace(vel, dt)
    return vel