from dataclasses import dataclass

import mink

from scripts.retargeting.tesollo_constants import (
    FINGERS,
    FINGER_PAIRS,
    COLLISION_PAIRS,
    SOLVER_NAME,
)


@dataclass
class TesolloTaskBundle:
    configuration: mink.Configuration
    posture_task: mink.PostureTask
    finger_tasks: list
    relative_finger_tasks: list
    wrist_task: mink.FrameTask
    tasks: list
    limits: list
    fingers: list
    finger_pairs: list
    solver: str


def build_tesollo_task_bundle(
    model,
    posture_cost: float = 1e-3,                # tiny — singularity regularizer only
    fingertip_position_cost: float = 100.0,    # dominant — drives fingertips to targets
    fingertip_orientation_cost: float = 0.5,
    fingertip_lm_damping: float = 0.1,
    relative_position_cost: float = 0.3,
    relative_orientation_cost: float = 0.0,
    relative_lm_damping: float = 1.0,
    wrist_position_cost: float = 250.0,
    wrist_orientation_cost: float = 40.0,
    wrist_lm_damping: float = 1.0,
    use_collision_avoidance: bool = False,
    geom_pairs: list | None = None,
    collision_min_distance: float = 0.002,
    collision_detection_distance: float = 0.01,
) -> TesolloTaskBundle:
    configuration = mink.Configuration(model)

    posture_task = mink.PostureTask(model, cost=posture_cost)

    finger_tasks = []
    for finger in FINGERS:
        task = mink.FrameTask(
            frame_name=finger,
            frame_type="geom",
            position_cost=fingertip_position_cost,
            orientation_cost=fingertip_orientation_cost,
            lm_damping=fingertip_lm_damping,
        )
        finger_tasks.append(task)

    relative_finger_tasks = []
    for idx1, idx2 in FINGER_PAIRS:
        task = mink.RelativeFrameTask(
            frame_name=FINGERS[idx1],
            frame_type="geom",
            root_name=FINGERS[idx2],
            root_type="geom",
            position_cost=relative_position_cost,
            orientation_cost=relative_orientation_cost,
            lm_damping=relative_lm_damping,
        )
        relative_finger_tasks.append(task)

    # Soft wrist pin: target set per frame to the tracked wrist (+ offset).
    # IK can deviate from this by mm/degrees if it helps reach fingertip targets.
    wrist_task = mink.FrameTask(
        frame_name="rh",
        frame_type="body",
        position_cost=wrist_position_cost,
        orientation_cost=wrist_orientation_cost,
        lm_damping=wrist_lm_damping,
    )

    limits = [mink.ConfigurationLimit(model=model)]
    if use_collision_avoidance:
        limits.append(
            mink.CollisionAvoidanceLimit(
                model=model,
                geom_pairs=geom_pairs if geom_pairs is not None else COLLISION_PAIRS,
                minimum_distance_from_collisions=collision_min_distance,
                collision_detection_distance=collision_detection_distance,
            )
        )

    tasks = [
        posture_task,
        wrist_task,
        *finger_tasks,
        *relative_finger_tasks,
    ]

    return TesolloTaskBundle(
        configuration=configuration,
        posture_task=posture_task,
        finger_tasks=finger_tasks,
        relative_finger_tasks=relative_finger_tasks,
        wrist_task=wrist_task,
        tasks=tasks,
        limits=limits,
        fingers=FINGERS,
        finger_pairs=FINGER_PAIRS,
        solver=SOLVER_NAME,
    )