"""CPU MuJoCo environment for batched MPPI rollouts.

Holds one MjModel and `num_samples` MjData instances. Each MjData carries
its own state and mocap fields, so per-sample wrist-orientation
perturbations are trivially supported.

Action layout (per step, per sample):
    a[:20]   — finger position-actuator targets (replaces data.ctrl)
    a[20:23] — wrist-orientation axis-angle delta, applied as
               mocap_quat = ref_wrist_quat ⊗ exp(delta)

The wrist *position* is always pinned to the reference (no sampling), and
the weld equality pulls the hand body to the mocap target. Cube is a true
free body driven by physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

# qpos layout in output_mocap_physics.xml: hand freejoint (7) + finger joints (20) + cube freejoint (7).
N_HAND_QPOS = 27
N_CUBE_QPOS = 7
N_FINGER_CTRL = 20
N_WRIST_DELTA = 3
N_ACTION = N_FINGER_CTRL + N_WRIST_DELTA  # 23

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]
TIP_BODIES = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]
CUBE_GEOM = "tracked_cube_geom"
CUBE_BODY = "tracked_cube"
WRIST_MOCAP = "wrist_mocap"


@dataclass
class EnvIds:
    """Resolved MuJoCo ids cached at setup."""
    wrist_mocap: int          # mocap index
    cube_qadr: int            # qpos index where cube freejoint starts
    cube_body_id: int
    cube_geom_id: int
    tip_body_ids: list[int]   # length 5
    tip_geom_ids: list[int]   # length 5


class CpuEnv:
    """One MjModel + N MjData rollout buffers + 1 outer 'world' MjData."""

    def __init__(self, xml_path: Path, num_samples: int):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.num_samples = num_samples
        self.world = mujoco.MjData(self.model)
        self.batch = [mujoco.MjData(self.model) for _ in range(num_samples)]
        self.ids = self._resolve_ids()
        if self.model.nq != N_HAND_QPOS + N_CUBE_QPOS:
            raise ValueError(
                f"Model nq={self.model.nq}, expected {N_HAND_QPOS + N_CUBE_QPOS}"
            )
        if self.model.nu != N_FINGER_CTRL:
            raise ValueError(f"Model nu={self.model.nu}, expected {N_FINGER_CTRL}")
        # Constant offset added to every finger ctrl (world + rollouts) so
        # position-controlled actuators apply real grip force around the IK
        # plan. Zero by default; set via set_closure_bias().
        self.ctrl_bias = np.zeros(N_FINGER_CTRL, dtype=np.float64)

    def set_closure_bias(self, bias_rad: float) -> None:
        """Set a constant flexion-direction offset added to every finger ctrl.
        Applies to both world.step and rollout_batch — the same effective
        actuator command is seen by predictions and the applied step.
        """
        self.ctrl_bias = bias_rad * self._flexion_closure_direction()
        # Clip to a sensible bound to keep ctrl within actuator ranges later.

    def _resolve_ids(self) -> EnvIds:
        m = self.model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
        wrist_mid = int(m.body_mocapid[wrist_bid])
        if wrist_mid < 0:
            raise ValueError("wrist_mocap is not a mocap body")
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        tip_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIP_BODIES]
        tip_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS]
        if any(i < 0 for i in (cube_qadr, cube_bid, cube_gid, *tip_bids, *tip_gids)):
            raise ValueError("Failed to resolve required ids in physics XML")
        return EnvIds(
            wrist_mocap=wrist_mid,
            cube_qadr=cube_qadr,
            cube_body_id=cube_bid,
            cube_geom_id=cube_gid,
            tip_body_ids=tip_bids,
            tip_geom_ids=tip_gids,
        )

    # ── state I/O ──────────────────────────────────────────────────────────

    def set_world_state(
        self,
        qpos: np.ndarray,
        cube_pose: np.ndarray,
        qvel: np.ndarray | None = None,
    ) -> None:
        """Initialise the outer world to a known kinematic state."""
        self.world.qpos[:N_HAND_QPOS] = qpos
        self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + N_CUBE_QPOS] = cube_pose
        if qvel is None:
            self.world.qvel[:] = 0.0
        else:
            self.world.qvel[:] = qvel
        self.world.mocap_pos[self.ids.wrist_mocap] = qpos[0:3]
        self.world.mocap_quat[self.ids.wrist_mocap] = qpos[3:7]
        mujoco.mj_forward(self.model, self.world)

    def _flexion_closure_direction(self) -> np.ndarray:
        """Per finger-joint closure direction (+1, -1, or 0 for abduction joints).

        Inferred from joint range: ranges starting at 0 close in +q direction,
        ranges ending at 0 close in -q direction, ranges straddling 0 are
        abduction-style and get no closure bias.
        """
        n_ctrl = N_FINGER_CTRL
        # Joints 7..27 in qpos correspond to ctrl 0..19 (1:1 since the only
        # other joint is the hand freejoint at qpos[:7]).
        # Use jnt_range to infer closure direction.
        d = np.zeros(n_ctrl, dtype=np.float64)
        for i in range(n_ctrl):
            # joint index = i+1 (skip hand freejoint at index 0)
            lo, hi = self.model.jnt_range[i + 1]
            if lo >= 0.0 and hi > 0.0:
                d[i] = 1.0
            elif hi <= 0.0 and lo < 0.0:
                d[i] = -1.0
            else:
                d[i] = 0.0   # abduction / straddles zero
        return d

    def warmup_grasp(
        self,
        cube_pose: np.ndarray,
        finger_ctrl: np.ndarray,
        ref_wrist_pos: np.ndarray,
        ref_wrist_quat: np.ndarray,
        n_steps: int,
        closure_bias: float = 0.0,
    ) -> None:
        """SPIDER-style virtual-stick initial grasp.

        Pin the cube at `cube_pose` and step physics with `finger_ctrl` so
        fingers settle into a stable grip before MPPI takes over. Cube qpos
        is overwritten and qvel zeroed after every substep, effectively
        welding the cube in place; finger PD response, contact friction
        normals, and collisions all develop naturally.

        `closure_bias` adds a per-joint offset toward the flexion direction
        on every finger ctrl during warmup, so position-controlled actuators
        actively *squeeze* into the cube rather than just settling at the IK
        position with zero force.
        """
        if n_steps <= 0:
            return
        cube_qadr = self.ids.cube_qadr
        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        cube_dofadr = int(self.model.jnt_dofadr[cube_jid])
        ctrl = np.asarray(finger_ctrl, dtype=np.float64).copy()
        if closure_bias > 0.0:
            ctrl += closure_bias * self._flexion_closure_direction()
            # Clamp to actuator control ranges so PD doesn't saturate weirdly.
            ctrl = np.clip(ctrl, self.model.actuator_ctrlrange[:, 0],
                                  self.model.actuator_ctrlrange[:, 1])
        self.world.qpos[cube_qadr : cube_qadr + N_CUBE_QPOS] = cube_pose
        self.world.qvel[cube_dofadr : cube_dofadr + 6] = 0.0
        self.world.ctrl[:] = ctrl
        self.world.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos
        self.world.mocap_quat[self.ids.wrist_mocap] = ref_wrist_quat
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.world)
            self.world.qpos[cube_qadr : cube_qadr + N_CUBE_QPOS] = cube_pose
            self.world.qvel[cube_dofadr : cube_dofadr + 6] = 0.0

    def snapshot_world_into_batch(self) -> None:
        """Copy world's (qpos, qvel) into every batch sample as the rollout init."""
        for d in self.batch:
            d.qpos[:] = self.world.qpos
            d.qvel[:] = self.world.qvel
            d.mocap_pos[:] = self.world.mocap_pos
            d.mocap_quat[:] = self.world.mocap_quat
            d.time = 0.0

    def step_world(self, action: np.ndarray, ref_wrist_pos: np.ndarray,
                   ref_wrist_quat: np.ndarray, n_substeps: int) -> None:
        """Advance the outer world by one MPC tick using `action`.

        action: (23,) — first 20 are finger ctrls, last 3 are wrist axis-angle delta
        ref_wrist_*: reference wrist pose this tick (mocap pos is locked to it)
        n_substeps: how many mj_step at sim_dt to make one MPC tick
        """
        ctrl = action[:N_FINGER_CTRL] + self.ctrl_bias
        ctrl = np.clip(ctrl, self.model.actuator_ctrlrange[:, 0],
                              self.model.actuator_ctrlrange[:, 1])
        delta_axisangle = action[N_FINGER_CTRL:]
        wrist_quat = quat_mul(ref_wrist_quat, axis_angle_to_quat(delta_axisangle))

        self.world.ctrl[:] = ctrl
        self.world.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos
        self.world.mocap_quat[self.ids.wrist_mocap] = wrist_quat
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.world)

    def rollout_batch(
        self,
        actions: np.ndarray,        # (N, H, 23)
        ref_wrist_pos: np.ndarray,  # (H, 3)
        ref_wrist_quat: np.ndarray, # (H, 4)
        n_substeps: int,
    ) -> np.ndarray:
        """Roll all N samples through the H-step horizon. Returns observations
        used by the reward function: dict of arrays, all shape (N, H, ...)."""
        N, H, _ = actions.shape
        obs = {
            "qpos": np.zeros((N, H, self.model.nq), dtype=np.float64),
            "qvel": np.zeros((N, H, self.model.nv), dtype=np.float64),
            "tip_xpos": np.zeros((N, H, 5, 3), dtype=np.float64),
            "cube_pos": np.zeros((N, H, 3), dtype=np.float64),
            "cube_quat": np.zeros((N, H, 4), dtype=np.float64),
            "tip_in_contact": np.zeros((N, H, 5), dtype=bool),
        }
        for i, d in enumerate(self.batch):
            for h in range(H):
                action = actions[i, h]
                ctrl = action[:N_FINGER_CTRL] + self.ctrl_bias
                ctrl = np.clip(ctrl, self.model.actuator_ctrlrange[:, 0],
                                      self.model.actuator_ctrlrange[:, 1])
                delta = action[N_FINGER_CTRL:]
                wrist_quat = quat_mul(ref_wrist_quat[h], axis_angle_to_quat(delta))
                d.ctrl[:] = ctrl
                d.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos[h]
                d.mocap_quat[self.ids.wrist_mocap] = wrist_quat
                for _ in range(n_substeps):
                    mujoco.mj_step(self.model, d)
                obs["qpos"][i, h] = d.qpos
                obs["qvel"][i, h] = d.qvel
                cube_qadr = self.ids.cube_qadr
                obs["cube_pos"][i, h] = d.qpos[cube_qadr : cube_qadr + 3]
                obs["cube_quat"][i, h] = d.qpos[cube_qadr + 3 : cube_qadr + 7]
                for f, tip_gid in enumerate(self.ids.tip_geom_ids):
                    obs["tip_xpos"][i, h, f] = d.geom_xpos[tip_gid]
                # Detect which tips are in contact with the cube this step.
                tip_gid_set = set(self.ids.tip_geom_ids)
                cube_gid = self.ids.cube_geom_id
                for c in range(d.ncon):
                    g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                    other = None
                    if g1 == cube_gid and g2 in tip_gid_set:
                        other = g2
                    elif g2 == cube_gid and g1 in tip_gid_set:
                        other = g1
                    if other is not None:
                        f = self.ids.tip_geom_ids.index(other)
                        obs["tip_in_contact"][i, h, f] = True
        return obs


# ── small math helpers ────────────────────────────────────────────────────────


def axis_angle_to_quat(axisangle: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (4,) quaternion (wxyz). Returns identity if magnitude is 0."""
    angle = float(np.linalg.norm(axisangle))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axisangle / angle
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both inputs/output wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz). Vectorised over leading dims."""
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    # 2 * cross(q.xyz, v)
    qv = np.concatenate([x, y, z], axis=-1)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)
