"""Multiprocessing-parallel CPU rollout for MPPI.

Each worker holds its own MjModel + MjData (loaded once via initializer),
and rolls out one sample per call. Pool is created once per session and
reused across MPC ticks. Reward is computed in-worker so only a scalar
crosses the pickle boundary per sample.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import mujoco
import numpy as np

from scripts.retargeting.dynamic_retargeting.cpu_env import (
    N_FINGER_CTRL, TIP_GEOMS, WRIST_MOCAP,
    axis_angle_to_quat, quat_mul,
)
from scripts.retargeting.dynamic_retargeting.reward import (
    RewardWeights, compute_horizon_reward,
)


# ── worker-local cache ────────────────────────────────────────────────────────


_W: dict = {}


def _init_worker(xml_path: str) -> None:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
    wrist_mid = int(model.body_mocapid[wrist_bid])
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qadr = int(model.jnt_qposadr[cube_jid])
    tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS]
    _W["model"] = model
    _W["data"] = data
    _W["wrist_mid"] = wrist_mid
    _W["cube_qadr"] = cube_qadr
    _W["tip_gids"] = tip_gids


def _rollout_one(args: tuple) -> float:
    (
        sample, ref_qpos, ref_cube_pos, ref_cube_quat,
        ref_wrist_pos, ref_wrist_quat,
        contact_mask, contact_target_cube, contact_target_valid,
        init_qpos, init_qvel, init_mocap_pos, init_mocap_quat,
        n_substeps, weights_tuple, ctrl_bias, contact_target_per_step,
    ) = args

    model = _W["model"]
    data = _W["data"]
    wrist_mid = _W["wrist_mid"]
    cube_qadr = _W["cube_qadr"]
    tip_gids = _W["tip_gids"]

    data.qpos[:] = init_qpos
    data.qvel[:] = init_qvel
    data.mocap_pos[:] = init_mocap_pos
    data.mocap_quat[:] = init_mocap_quat
    data.time = 0.0

    H = sample.shape[0]
    obs_qpos = np.zeros((H, model.nq), dtype=np.float64)
    obs_cube_pos = np.zeros((H, 3), dtype=np.float64)
    obs_cube_quat = np.zeros((H, 4), dtype=np.float64)
    obs_tip_xpos = np.zeros((H, 5, 3), dtype=np.float64)

    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]
    for h in range(H):
        action = sample[h]
        finger_ctrl = np.clip(action[:N_FINGER_CTRL] + ctrl_bias, ctrl_lo, ctrl_hi)
        delta = action[N_FINGER_CTRL:]
        wrist_quat = quat_mul(ref_wrist_quat[h], axis_angle_to_quat(delta))
        data.ctrl[:] = finger_ctrl
        data.mocap_pos[wrist_mid] = ref_wrist_pos[h]
        data.mocap_quat[wrist_mid] = wrist_quat
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        obs_qpos[h] = data.qpos
        obs_cube_pos[h] = data.qpos[cube_qadr : cube_qadr + 3]
        obs_cube_quat[h] = data.qpos[cube_qadr + 3 : cube_qadr + 7]
        for f, gid in enumerate(tip_gids):
            obs_tip_xpos[h, f] = data.geom_xpos[gid]

    # weights_tuple = (cube_pos, cube_quat, finger_qpos, contact, wrist_delta, contact_always_on)
    weights = RewardWeights(*weights_tuple)
    obs = {
        "qpos": obs_qpos[None, :, :],
        "cube_pos": obs_cube_pos[None, :, :],
        "cube_quat": obs_cube_quat[None, :, :],
        "tip_xpos": obs_tip_xpos[None, :, :, :],
    }
    rews = compute_horizon_reward(
        obs=obs,
        ref_qpos=ref_qpos,
        ref_cube_pos=ref_cube_pos,
        ref_cube_quat=ref_cube_quat,
        ref_wrist_quat=ref_wrist_quat,
        contact_mask=contact_mask,
        contact_target_cube=contact_target_cube,
        contact_target_valid=contact_target_valid,
        actions=sample[None, :, :],
        weights=weights,
        contact_target_per_step=contact_target_per_step,
    )
    return float(rews[0])


# ── pool wrapper ──────────────────────────────────────────────────────────────


class MpCpuEnv:
    def __init__(self, xml_path: Path, num_workers: int | None = None):
        self.xml_path = Path(xml_path)
        self.num_workers = num_workers or os.cpu_count() or 1
        ctx = mp.get_context("fork")
        self.pool = ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(str(self.xml_path),),
        )

    def score_samples(
        self,
        samples: np.ndarray,             # (N, H, 23)
        ref_qpos: np.ndarray,            # (H, 27)
        ref_cube_pos: np.ndarray,        # (H, 3)
        ref_cube_quat: np.ndarray,       # (H, 4)
        ref_wrist_pos: np.ndarray,       # (H, 3)
        ref_wrist_quat: np.ndarray,      # (H, 4)
        contact_mask: np.ndarray,        # (5,)
        contact_target_cube: np.ndarray, # (5, 3)
        contact_target_valid: np.ndarray,# (5,)
        init_qpos: np.ndarray,
        init_qvel: np.ndarray,
        init_mocap_pos: np.ndarray,
        init_mocap_quat: np.ndarray,
        n_substeps: int,
        weights: RewardWeights,
        ctrl_bias: np.ndarray | None = None,
        contact_target_per_step: np.ndarray | None = None,
    ) -> np.ndarray:
        wt = (
            weights.cube_pos, weights.cube_quat, weights.finger_qpos,
            weights.contact, weights.wrist_delta, weights.contact_always_on,
        )
        if ctrl_bias is None:
            ctrl_bias = np.zeros(N_FINGER_CTRL, dtype=np.float64)
        N = samples.shape[0]
        args_list = [
            (
                samples[i], ref_qpos, ref_cube_pos, ref_cube_quat,
                ref_wrist_pos, ref_wrist_quat,
                contact_mask, contact_target_cube, contact_target_valid,
                init_qpos, init_qvel, init_mocap_pos, init_mocap_quat,
                n_substeps, wt, ctrl_bias, contact_target_per_step,
            )
            for i in range(N)
        ]
        rews = self.pool.map(_rollout_one, args_list)
        return np.asarray(rews, dtype=np.float64)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
