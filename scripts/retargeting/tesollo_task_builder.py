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
    tasks: list
    limits: list
    fingers: list
    finger_pairs: list
    solver: str


def build_tesollo_task_bundle(
    model,
    posture_cost: float = 5e-2,
    fingertip_position_cost: float = 1.0,
    fingertip_orientation_cost: float = 0.01,
    fingertip_lm_damping: float = 0.1,        # was 1.0 — lower → frame task more responsive
    relative_position_cost: float = 0.3,      # was 1.0 — was dominating frame tasks (10 pairs vs 5 fingers)
    relative_orientation_cost: float = 0.0,
    relative_lm_damping: float = 1.0,         # was 0.1 — higher → relative task more compliant (regularizer)
    use_collision_avoidance: bool = False,
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

    limits = [mink.ConfigurationLimit(model=model)]
    if use_collision_avoidance:
        limits.append(
            mink.CollisionAvoidanceLimit(
                model=model,
                geom_pairs=COLLISION_PAIRS,
            )
        )

    tasks = [
        posture_task,
        *finger_tasks,
        *relative_finger_tasks,
    ]

    return TesolloTaskBundle(
        configuration=configuration,
        posture_task=posture_task,
        finger_tasks=finger_tasks,
        relative_finger_tasks=relative_finger_tasks,
        tasks=tasks,
        limits=limits,
        fingers=FINGERS,
        finger_pairs=FINGER_PAIRS,
        solver=SOLVER_NAME,
    )