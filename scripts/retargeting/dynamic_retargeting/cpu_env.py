"""CPU MuJoCo environment for Stage 3 sampling MPC.

Loads the *dyn* scene from `build_ik_scene.compile_dyn`: cube freejoint is
replaced by 3 slide + 3 hinge joints with position-style actuators (the
"virtual stick"). Action vector is 29-dim per step:

    a[ 0:20] — finger position-actuator targets (replaces data.ctrl[:20])
    a[20:23] — wrist-orientation axis-angle delta (mocap rotation perturb)
    a[23:26] — cube position target (slide actuator ctrl)
    a[26:29] — cube rotation target (hinge actuator ctrl, intrinsic XYZ Euler)

Wrist *position* mocap is always pinned to the reference. The actuator gain
on the six cube DOFs is scaled at runtime via `set_obj_decay` to implement
the virtual-stick decay schedule (early MPC ticks: stick on, fingers free
to settle; late ticks: stick off, fingers must hold).

Two environment objects:
    `world`  — one MjData advanced once per MPC tick with the chosen plan.
    `batch`  — N MjData for sampled rollouts, stepped horizon-by-horizon so
               we can interleave resample-on-drop swaps between sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.retargeting.build_ik_scene import compile_dyn, scale_obj_actuators


# Action / state layout constants — kept here so mppi.py and reward.py
# can reuse them without knowing about MjSpec specifics.
N_HAND_QPOS = 27                 # wrist freejoint (7) + 20 finger joints
N_CUBE_QPOS = 6                  # cube_pos_xyz + cube_rot_xyz (intrinsic XYZ)

N_FINGER_CTRL = 20
N_WRIST_DELTA = 3
N_OBJ_POS_CTRL = 3
N_OBJ_ROT_CTRL = 3
N_OBJ_CTRL = N_OBJ_POS_CTRL + N_OBJ_ROT_CTRL
N_ACTION = N_FINGER_CTRL + N_WRIST_DELTA + N_OBJ_CTRL  # 29

# Slice helpers — index a (..., A) action array.
SLICE_FINGER = slice(0, N_FINGER_CTRL)
SLICE_WRIST = slice(N_FINGER_CTRL, N_FINGER_CTRL + N_WRIST_DELTA)
SLICE_OBJ_POS = slice(N_FINGER_CTRL + N_WRIST_DELTA,
                      N_FINGER_CTRL + N_WRIST_DELTA + N_OBJ_POS_CTRL)
SLICE_OBJ_ROT = slice(N_FINGER_CTRL + N_WRIST_DELTA + N_OBJ_POS_CTRL,
                      N_ACTION)

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]
TIP_BODIES = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]
CUBE_GEOM = "tracked_cube_geom"
WRIST_MOCAP = "wrist_mocap"


@dataclass
class EnvIds:
    wrist_mocap: int
    cube_qadr: int               # qpos start of the 6 cube DOFs
    cube_dofadr: int             # qvel start of the 6 cube DOFs
    cube_geom_id: int
    tip_geom_ids: list[int]
    finger_act_lo: np.ndarray    # (20,)  ctrlrange[:, 0] for fingers
    finger_act_hi: np.ndarray    # (20,)


def quat_to_euler_xyz_intrinsic(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion → 3 intrinsic XYZ Euler angles matching the
    dyn scene joint chain (cube_rot_x, cube_rot_y, cube_rot_z in body frame).
    """
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return R.from_quat(quat_xyzw).as_euler("XYZ", degrees=False)


def euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    """Inverse of `quat_to_euler_xyz_intrinsic` for diagnostics / reward."""
    q_xyzw = R.from_euler("XYZ", euler_xyz, degrees=False).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def axis_angle_to_quat_wxyz(axisangle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axisangle))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axisangle / angle
    s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def cube_qpos_from_pose7(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz]  →  (6,) cube_qpos for the dyn scene."""
    pos = np.asarray(pose7[:3], dtype=float)
    euler = quat_to_euler_xyz_intrinsic(np.asarray(pose7[3:7], dtype=float))
    return np.concatenate([pos, euler])


def cube_pose7_from_qpos(cube_qpos: np.ndarray) -> np.ndarray:
    """(6,) dyn cube_qpos → [x,y,z,qw,qx,qy,qz] pose vector."""
    pos = cube_qpos[:3]
    quat = euler_xyz_to_quat_wxyz(cube_qpos[3:6])
    return np.concatenate([pos, quat])


class CpuEnv:
    """Sequential CPU rollout env. Owns 1 world + N batch MjData."""

    def __init__(self, num_samples: int, sim_dt: float = 0.002):
        self.model, _ = compile_dyn(sim_dt=sim_dt)
        self.num_samples = num_samples
        self.world = mujoco.MjData(self.model)
        self.batch = [mujoco.MjData(self.model) for _ in range(num_samples)]
        self.ids = self._resolve_ids()
        if self.model.nq != N_HAND_QPOS + N_CUBE_QPOS:
            raise ValueError(f"Model nq={self.model.nq}, expected "
                             f"{N_HAND_QPOS + N_CUBE_QPOS}")
        if self.model.nu != N_FINGER_CTRL + N_OBJ_CTRL:
            raise ValueError(f"Model nu={self.model.nu}, expected "
                             f"{N_FINGER_CTRL + N_OBJ_CTRL}")
        self.ctrl_bias = np.zeros(N_FINGER_CTRL, dtype=np.float64)
        self.obj_decay = 1.0  # current virtual-stick scale (full = 1, off = 0)

    def _resolve_ids(self) -> EnvIds:
        m = self.model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
        wrist_mid = int(m.body_mocapid[wrist_bid])
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_pos_x")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        cube_dofadr = int(m.jnt_dofadr[cube_jid])
        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        tip_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS]
        if any(i < 0 for i in (wrist_mid, cube_qadr, cube_dofadr, cube_gid, *tip_gids)):
            raise ValueError("Failed to resolve required ids in dyn scene")
        # Finger actuators are the first 20 (defined before the 6 cube actuators).
        finger_lo = m.actuator_ctrlrange[:N_FINGER_CTRL, 0].copy()
        finger_hi = m.actuator_ctrlrange[:N_FINGER_CTRL, 1].copy()
        return EnvIds(
            wrist_mocap=wrist_mid,
            cube_qadr=cube_qadr,
            cube_dofadr=cube_dofadr,
            cube_geom_id=cube_gid,
            tip_geom_ids=tip_gids,
            finger_act_lo=finger_lo,
            finger_act_hi=finger_hi,
        )

    # ── virtual-stick scaling ─────────────────────────────────────────────

    def set_obj_decay(self, factor: float) -> None:
        """Scale the 6 cube actuator gains by `factor` (0…1)."""
        self.obj_decay = float(factor)
        scale_obj_actuators(self.model, factor)

    def set_closure_bias(self, bias_rad: float) -> None:
        self.ctrl_bias = bias_rad * self._flexion_closure_direction()

    def _flexion_closure_direction(self) -> np.ndarray:
        d = np.zeros(N_FINGER_CTRL, dtype=np.float64)
        for i in range(N_FINGER_CTRL):
            lo, hi = self.model.jnt_range[i + 1]   # +1 skips hand freejoint
            if lo >= 0.0 and hi > 0.0:
                d[i] = 1.0
            elif hi <= 0.0 and lo < 0.0:
                d[i] = -1.0
        return d

    # ── world I/O ─────────────────────────────────────────────────────────

    def set_world_state(self, hand_qpos: np.ndarray, cube_pose7: np.ndarray,
                        qvel: np.ndarray | None = None) -> None:
        cube_q = cube_qpos_from_pose7(cube_pose7)
        self.world.qpos[:N_HAND_QPOS] = hand_qpos
        self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + N_CUBE_QPOS] = cube_q
        self.world.qvel[:] = 0.0 if qvel is None else qvel
        self.world.mocap_pos[self.ids.wrist_mocap] = hand_qpos[0:3]
        self.world.mocap_quat[self.ids.wrist_mocap] = hand_qpos[3:7]
        mujoco.mj_forward(self.model, self.world)

    def get_world_cube_pose7(self) -> np.ndarray:
        cube_q = self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + N_CUBE_QPOS]
        return cube_pose7_from_qpos(cube_q)

    def get_world_cube_pos(self) -> np.ndarray:
        return self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + 3].copy()

    # ── action application ────────────────────────────────────────────────

    def _apply_action_to_data(self, data: mujoco.MjData, action: np.ndarray,
                              ref_wrist_pos: np.ndarray,
                              ref_wrist_quat: np.ndarray) -> None:
        finger_ctrl = np.clip(action[SLICE_FINGER] + self.ctrl_bias,
                              self.ids.finger_act_lo, self.ids.finger_act_hi)
        delta = action[SLICE_WRIST]
        wrist_quat = quat_mul_wxyz(ref_wrist_quat, axis_angle_to_quat_wxyz(delta))
        data.ctrl[:N_FINGER_CTRL] = finger_ctrl
        data.ctrl[N_FINGER_CTRL : N_FINGER_CTRL + N_OBJ_POS_CTRL] = action[SLICE_OBJ_POS]
        data.ctrl[N_FINGER_CTRL + N_OBJ_POS_CTRL : N_FINGER_CTRL + N_OBJ_CTRL] = action[SLICE_OBJ_ROT]
        data.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos
        data.mocap_quat[self.ids.wrist_mocap] = wrist_quat

    def step_world(self, action: np.ndarray, ref_wrist_pos: np.ndarray,
                   ref_wrist_quat: np.ndarray, n_substeps: int) -> None:
        self._apply_action_to_data(self.world, action, ref_wrist_pos, ref_wrist_quat)
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.world)

    # ── batched rollout ───────────────────────────────────────────────────

    def snapshot_world_into_batch(self) -> None:
        for d in self.batch:
            d.qpos[:] = self.world.qpos
            d.qvel[:] = self.world.qvel
            d.ctrl[:] = self.world.ctrl
            d.mocap_pos[:] = self.world.mocap_pos
            d.mocap_quat[:] = self.world.mocap_quat
            d.time = 0.0

    def rollout_with_resample(
        self,
        actions: np.ndarray,            # (N, H, 29)  — MAY be modified in place
        ref_wrist_pos: np.ndarray,      # (H, 3)
        ref_wrist_quat: np.ndarray,     # (H, 4)
        n_substeps: int,
        ref_cube_pos: np.ndarray,       # (H, 3)
        drop_threshold_m: float | None = None,
    ) -> dict:
        """Roll out all N samples through the H-step horizon, stepping
        horizon-by-horizon so we can swap dropped-cube samples with good
        ones between steps (SPIDER resample-on-drop). Returns observations
        used by the reward function plus per-sample terminate flag.
        """
        N = self.num_samples
        H = actions.shape[1]
        obs = {
            "qpos": np.zeros((N, H, self.model.nq), dtype=np.float64),
            "cube_pos": np.zeros((N, H, 3), dtype=np.float64),
            "cube_quat_wxyz": np.zeros((N, H, 4), dtype=np.float64),
            "tip_xpos": np.zeros((N, H, 5, 3), dtype=np.float64),
            "tip_in_contact": np.zeros((N, H, 5), dtype=bool),
        }
        terminated = np.zeros(N, dtype=bool)
        rng = np.random.default_rng()
        cube_gid = self.ids.cube_geom_id
        tip_gids = self.ids.tip_geom_ids
        tip_gid_set = set(tip_gids)
        cube_qadr = self.ids.cube_qadr

        for h in range(H):
            ref_wpos_h = ref_wrist_pos[h]
            ref_wquat_h = ref_wrist_quat[h]
            for i in range(N):
                d = self.batch[i]
                self._apply_action_to_data(d, actions[i, h], ref_wpos_h, ref_wquat_h)
                for _ in range(n_substeps):
                    mujoco.mj_step(self.model, d)
                # observation
                obs["qpos"][i, h] = d.qpos
                obs["cube_pos"][i, h] = d.qpos[cube_qadr : cube_qadr + 3]
                obs["cube_quat_wxyz"][i, h] = euler_xyz_to_quat_wxyz(
                    d.qpos[cube_qadr + 3 : cube_qadr + 6])
                for f, gid in enumerate(tip_gids):
                    obs["tip_xpos"][i, h, f] = d.geom_xpos[gid]
                for c in range(d.ncon):
                    g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
                    other = None
                    if g1 == cube_gid and g2 in tip_gid_set:
                        other = g2
                    elif g2 == cube_gid and g1 in tip_gid_set:
                        other = g1
                    if other is not None:
                        obs["tip_in_contact"][i, h, tip_gids.index(other)] = True

            # Resample-on-drop between horizon steps: replace samples whose
            # cube has slipped beyond `drop_threshold_m` of the reference by
            # cloning a good sample's state + ctrl history. Doesn't run on
            # the last step (no future steps would benefit).
            if (drop_threshold_m is not None
                    and h < H - 1
                    and not terminated.all()):
                cube_pos_err = np.linalg.norm(
                    obs["cube_pos"][:, h] - ref_cube_pos[h], axis=-1)
                dropped = (cube_pos_err > drop_threshold_m) & ~terminated
                good = ~dropped & ~terminated
                if dropped.any() and good.any():
                    bad_idx = np.where(dropped)[0]
                    good_idx = np.where(good)[0]
                    picks = rng.choice(good_idx, size=bad_idx.shape[0], replace=True)
                    for bi, gi in zip(bad_idx, picks):
                        bd, gd = self.batch[bi], self.batch[gi]
                        bd.qpos[:] = gd.qpos
                        bd.qvel[:] = gd.qvel
                        bd.ctrl[:] = gd.ctrl
                        bd.mocap_pos[:] = gd.mocap_pos
                        bd.mocap_quat[:] = gd.mocap_quat
                        # Copy ctrl history through current step so downstream
                        # reward sees the *applied* sequence.
                        actions[bi, : h + 1] = actions[gi, : h + 1]
                        # Copy observations through current step too — keeps
                        # per-sample obs trace internally consistent.
                        for key in obs:
                            obs[key][bi, : h + 1] = obs[key][gi, : h + 1]
                elif dropped.all():
                    terminated |= dropped  # nothing to copy from; let them be

        obs["terminated"] = terminated
        return obs
