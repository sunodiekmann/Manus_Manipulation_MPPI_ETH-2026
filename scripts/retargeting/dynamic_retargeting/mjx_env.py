"""MJX (JAX-vmapped) rollout + reward for MPPI.

The whole inner loop — N samples × H horizon steps × n_substeps physics steps
plus reward — is one jitted function, vmapped over samples. Runs on GPU when
JAX is on cuda; same code works on CPU when forced.

Interface mirrors CpuEnv just enough for mppi.py to dispatch on backend.
The reward formulation matches `reward.py` (replicated in jnp here so the
whole thing stays inside JIT).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CUBE_GEOM, CUBE_BODY, FINGER_NAMES, TIP_GEOMS, WRIST_MOCAP,
    N_HAND_QPOS, N_CUBE_QPOS, N_FINGER_CTRL, N_ACTION,
)
from scripts.retargeting.dynamic_retargeting.reward import RewardWeights


# ── jnp math helpers ──────────────────────────────────────────────────────────


def _axis_angle_to_quat_jnp(axisangle: jnp.ndarray) -> jnp.ndarray:
    """(3,) → (4,) wxyz. Uses sinc to stay smooth at zero magnitude."""
    angle = jnp.linalg.norm(axisangle)
    half = 0.5 * angle
    # sinc-based form: q = [cos(half), axisangle * sinc(half) / 2]
    safe = jnp.where(angle < 1e-9, 1.0, angle)
    sin_half = jnp.sin(half)
    s = jnp.where(angle < 1e-9, 0.5, sin_half / safe)
    return jnp.array([jnp.cos(half), axisangle[0] * s, axisangle[1] * s, axisangle[2] * s])


def _quat_mul_jnp(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_rotate_vec_jnp(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """q: (..., 4), v: (..., 3). Output (..., 3)."""
    w = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * jnp.cross(qv, v, axis=-1)
    return v + w * t + jnp.cross(qv, t, axis=-1)


# ── env ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MjxIds:
    wrist_mocap: int
    cube_qadr: int
    tip_gids: tuple


class MjxEnv:
    """Holds the MJX model + an initial mjx.Data, plus jit-compiled rollout."""

    def __init__(self, xml_path: Path, n_substeps: int, sim_dt: float | None = None):
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        if sim_dt is not None:
            # Override XML timestep for compile-time savings. Cube is light/slow,
            # 5–10× the default 0.002 is stable.
            self.mj_model.opt.timestep = float(sim_dt)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.init_mjx_data = mjx.make_data(self.mjx_model)
        self.n_substeps = int(n_substeps)

        m = self.mj_model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
        wrist_mid = int(m.body_mocapid[wrist_bid])
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        tip_gids = tuple(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS
        )
        self.ids = MjxIds(wrist_mocap=wrist_mid, cube_qadr=cube_qadr, tip_gids=tip_gids)

        # ── pre-jit the rollout-and-score kernel ──
        self._rollout_jit = jax.jit(
            self._build_rollout(),
            static_argnames=(),
        )
        self._mj_step_per_substep = self.n_substeps

    # ----------------------------------------------------------------------

    def _build_rollout(self):
        """Return a closure: (init_data, samples, refs, contact, weights_arr) → rewards (N,)."""
        n_substeps = self.n_substeps
        wrist_mid = self.ids.wrist_mocap
        cube_qadr = self.ids.cube_qadr
        tip_gids = jnp.array(self.ids.tip_gids, dtype=jnp.int32)
        mjx_model = self.mjx_model

        def _multi_step(data):
            # Python unroll: small constant n_substeps compiles faster than fori_loop.
            for _ in range(n_substeps):
                data = mjx.step(mjx_model, data)
            return data

        def _step(carry, inputs):
            """One MPC tick for one sample."""
            data = carry
            action, ref_wp, ref_wq = inputs
            finger_ctrl = action[:N_FINGER_CTRL]
            wrist_delta = action[N_FINGER_CTRL:]
            wrist_quat = _quat_mul_jnp(ref_wq, _axis_angle_to_quat_jnp(wrist_delta))
            data = data.replace(
                ctrl=finger_ctrl,
                mocap_pos=data.mocap_pos.at[wrist_mid].set(ref_wp),
                mocap_quat=data.mocap_quat.at[wrist_mid].set(wrist_quat),
            )
            data = _multi_step(data)
            cube_pos = data.qpos[cube_qadr : cube_qadr + 3]
            cube_quat = data.qpos[cube_qadr + 3 : cube_qadr + 7]
            finger_qpos = data.qpos[7 : 27]
            tip_xpos = data.geom_xpos[tip_gids]   # (5, 3)
            return data, (cube_pos, cube_quat, finger_qpos, tip_xpos)

        def _reward(traj, sample_actions,
                    ref_qpos_h, ref_cube_pos_h, ref_cube_quat_h,
                    contact_mask, contact_target_cube, contact_target_valid,
                    w_cp, w_cq, w_fq, w_ct, w_wd):
            cube_pos, cube_quat, finger_qpos, tip_xpos = traj  # each (H, ...)
            pos_err2 = jnp.sum((cube_pos - ref_cube_pos_h) ** 2, axis=-1)
            dot = jnp.sum(cube_quat * ref_cube_quat_h, axis=-1)
            quat_err = 1.0 - dot ** 2
            finger_err2 = jnp.sum((finger_qpos - ref_qpos_h[:, 7:27]) ** 2, axis=-1)
            target_world = (
                ref_cube_pos_h[:, None, :]
                + _quat_rotate_vec_jnp(
                    ref_cube_quat_h[:, None, :],
                    contact_target_cube[None, :, :],
                )
            )                                                   # (H, 5, 3)
            contact_err = jnp.sum((tip_xpos - target_world) ** 2, axis=-1)
            wpf = (contact_mask & contact_target_valid).astype(jnp.float32)
            contact_term = jnp.sum(contact_err * wpf[None, :], axis=-1)
            wrist_pen = jnp.sum(sample_actions[:, N_FINGER_CTRL:] ** 2, axis=-1)
            per_step = (
                -w_cp * pos_err2
                - w_cq * quat_err
                - w_fq * finger_err2
                - w_ct * contact_term
                - w_wd * wrist_pen
            )
            return per_step.sum()

        def _per_sample(init_data, sample_actions,
                        ref_qpos_h, ref_cube_pos_h, ref_cube_quat_h,
                        ref_wrist_pos_h, ref_wrist_quat_h,
                        contact_mask, contact_target_cube, contact_target_valid,
                        w_cp, w_cq, w_fq, w_ct, w_wd):
            _, traj = jax.lax.scan(
                _step, init_data,
                (sample_actions, ref_wrist_pos_h, ref_wrist_quat_h),
            )
            return _reward(traj, sample_actions,
                           ref_qpos_h, ref_cube_pos_h, ref_cube_quat_h,
                           contact_mask, contact_target_cube, contact_target_valid,
                           w_cp, w_cq, w_fq, w_ct, w_wd)

        # vmap over samples (axis 0 of sample_actions)
        rollout_batch = jax.vmap(
            _per_sample,
            in_axes=(None, 0,
                     None, None, None,
                     None, None,
                     None, None, None,
                     None, None, None, None, None),
        )

        return rollout_batch

    # ----------------------------------------------------------------------

    def rollout_and_score(
        self,
        init_qpos: np.ndarray,           # (nq,)
        init_qvel: np.ndarray,           # (nv,)
        init_mocap_pos: np.ndarray,      # (n_mocap, 3)
        init_mocap_quat: np.ndarray,     # (n_mocap, 4)
        samples: np.ndarray,             # (N, H, 23)
        ref_qpos_h: np.ndarray,          # (H, 27)
        ref_cube_pos_h: np.ndarray,      # (H, 3)
        ref_cube_quat_h: np.ndarray,     # (H, 4)
        ref_wrist_pos_h: np.ndarray,     # (H, 3)
        ref_wrist_quat_h: np.ndarray,    # (H, 4)
        contact_mask: np.ndarray,        # (5,)
        contact_target_cube: np.ndarray, # (5, 3)
        contact_target_valid: np.ndarray,# (5,)
        weights: RewardWeights,
    ) -> np.ndarray:
        """Returns rewards (N,) on host."""
        init_data = self.init_mjx_data.replace(
            qpos=jnp.asarray(init_qpos),
            qvel=jnp.asarray(init_qvel),
            mocap_pos=jnp.asarray(init_mocap_pos),
            mocap_quat=jnp.asarray(init_mocap_quat),
        )
        # mj_forward to populate site/geom xpos before rollout starts
        init_data = mjx.forward(self.mjx_model, init_data)
        rews = self._rollout_jit(
            init_data,
            jnp.asarray(samples),
            jnp.asarray(ref_qpos_h),
            jnp.asarray(ref_cube_pos_h),
            jnp.asarray(ref_cube_quat_h),
            jnp.asarray(ref_wrist_pos_h),
            jnp.asarray(ref_wrist_quat_h),
            jnp.asarray(contact_mask, dtype=bool),
            jnp.asarray(contact_target_cube),
            jnp.asarray(contact_target_valid, dtype=bool),
            float(weights.cube_pos),
            float(weights.cube_quat),
            float(weights.finger_qpos),
            float(weights.contact),
            float(weights.wrist_delta),
        )
        return np.asarray(rews)
