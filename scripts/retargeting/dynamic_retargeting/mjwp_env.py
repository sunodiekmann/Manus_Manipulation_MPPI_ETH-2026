"""GPU-batched rollout for Stage 3 sampling MPC, on MuJoCo Warp.

Mirrors `cpu_env.CpuEnv` semantics — same dyn scene (cube as a real
freejoint, no virtual stick), same 23-dim action layout, same
`rollout_with_resample` signature — but with N worlds stepped in parallel
via NVIDIA Warp CUDA kernels.

Host↔device shuffling per MPC tick is small (action knots in, cube state
out for resample-on-drop, reward computed host-side). First call compiles
+ caches Warp kernels on disk; subsequent runs are fast.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjwp
import numpy as np
import warp as wp

from scripts.retargeting.build_ik_scene import compile_dyn
from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CUBE_GEOM,
    CUBE_FREEJOINT,
    DEFAULT_SIM_DT,
    N_ACTION,
    N_CUBE_QPOS,
    N_FINGER_CTRL,
    N_HAND_QPOS,
    N_WRIST_DELTA,
    SLICE_FINGER,
    SLICE_WRIST,
    TIP_GEOMS,
    WRIST_MOCAP,
    apply_spider_physics,
    axis_angle_to_quat_wxyz,
    quat_mul_wxyz,
    slerp_wxyz,
)


def _batched_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Vectorised Hamilton product. q1, q2: (..., 4) wxyz."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _batched_axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """(N, 3) axis-angle → (N, 4) wxyz quaternion."""
    angle = np.linalg.norm(aa, axis=-1)
    half = 0.5 * angle
    safe = np.where(angle < 1e-9, 1.0, angle)
    s = np.where(angle < 1e-9, 0.5, np.sin(half) / safe)
    q = np.empty(aa.shape[:-1] + (4,), dtype=aa.dtype)
    q[..., 0] = np.cos(half)
    q[..., 1] = aa[..., 0] * s
    q[..., 2] = aa[..., 1] * s
    q[..., 3] = aa[..., 2] * s
    return q


class MjwpEnv:
    """Same surface as cpu_env.CpuEnv (set_world_state, step_world,
    snapshot_world_into_batch, rollout_with_resample, warmup_grasp,
    get_world_cube_*), but stepping N worlds in parallel on GPU via
    mujoco_warp.
    """

    def __init__(self, num_samples: int, sim_dt: float = DEFAULT_SIM_DT,
                 spider_physics: bool = True):
        self.mj_model, _ = compile_dyn(sim_dt=sim_dt)
        self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.mj_model.opt.iterations = 20
        self.mj_model.opt.ls_iterations = 50
        if spider_physics:
            apply_spider_physics(self.mj_model)
        if self.mj_model.nu != N_FINGER_CTRL:
            raise ValueError(f"Model nu={self.mj_model.nu}, expected "
                             f"{N_FINGER_CTRL}")
        if self.mj_model.nq != N_HAND_QPOS + N_CUBE_QPOS:
            raise ValueError(f"Model nq={self.mj_model.nq}, expected "
                             f"{N_HAND_QPOS + N_CUBE_QPOS}")
        self.num_samples = int(num_samples)
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu

        m = self.mj_model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
        self.wrist_mocap_idx = int(m.body_mocapid[wrist_bid])
        self.nmocap = int(m.nmocap)
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, CUBE_FREEJOINT)
        self.cube_qadr = int(m.jnt_qposadr[cube_jid])
        self.cube_dofadr = int(m.jnt_dofadr[cube_jid])
        self.tip_gids = tuple(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS)
        if any(g < 0 for g in self.tip_gids):
            raise ValueError("tip geom resolution failed in dyn scene")
        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        if cube_gid < 0:
            raise ValueError(f"cube geom '{CUBE_GEOM}' not found")
        self.cube_gid = cube_gid
        self.finger_act_lo = m.actuator_ctrlrange[:N_FINGER_CTRL, 0].copy()
        self.finger_act_hi = m.actuator_ctrlrange[:N_FINGER_CTRL, 1].copy()

        # Two parallel Data objects: nworld=1 for the outer world and
        # nworld=N for the rollout batch. Both share the same Model.
        self.mw = mjwp.put_model(m)
        self.dw_world = mjwp.make_data(m, nworld=1, njmax=256)
        self.dw_batch = mjwp.make_data(m, nworld=self.num_samples, njmax=256)

        # CUDA Graph capture for both steppers.
        mjwp.step(self.mw, self.dw_world); wp.synchronize()
        mjwp.step(self.mw, self.dw_batch); wp.synchronize()
        with wp.ScopedCapture() as cap_w:
            mjwp.step(self.mw, self.dw_world)
        wp.synchronize()
        self._graph_world = cap_w.graph
        with wp.ScopedCapture() as cap_b:
            mjwp.step(self.mw, self.dw_batch)
        wp.synchronize()
        self._graph_batch = cap_b.graph
        # Host-side mirrors of the world state.
        self._world_qpos = np.zeros(self.nq, dtype=np.float64)
        self._world_qvel = np.zeros(self.nv, dtype=np.float64)
        self._world_mocap_pos = np.zeros((self.nmocap, 3), dtype=np.float64)
        self._world_mocap_quat = np.zeros((self.nmocap, 4), dtype=np.float64)
        self._world_mocap_quat[:, 0] = 1.0
        self._world_ctrl = np.zeros(self.nu, dtype=np.float64)

    # ── ergonomic API mirror of CpuEnv ────────────────────────────────────

    def set_world_state(self, hand_qpos: np.ndarray, cube_pose7: np.ndarray,
                        qvel: np.ndarray | None = None) -> None:
        self._world_qpos[:N_HAND_QPOS] = hand_qpos
        self._world_qpos[self.cube_qadr : self.cube_qadr + N_CUBE_QPOS] = cube_pose7
        self._world_qvel[:] = 0.0 if qvel is None else qvel
        self._world_mocap_pos[self.wrist_mocap_idx] = hand_qpos[0:3]
        self._world_mocap_quat[self.wrist_mocap_idx] = hand_qpos[3:7]
        self._world_ctrl[:] = 0.0
        self._push_world_state()

    def _push_world_state(self) -> None:
        """Mirror the host-side world state into the nworld=1 Data."""
        self.dw_world.qpos.assign(self._world_qpos[None].astype(np.float32))
        self.dw_world.qvel.assign(self._world_qvel[None].astype(np.float32))
        self.dw_world.ctrl.assign(self._world_ctrl[None].astype(np.float32))
        self.dw_world.mocap_pos.assign(self._world_mocap_pos[None].astype(np.float32))
        self.dw_world.mocap_quat.assign(self._world_mocap_quat[None].astype(np.float32))

    def get_world_cube_pos(self) -> np.ndarray:
        return self._world_qpos[self.cube_qadr : self.cube_qadr + 3].copy()

    def get_world_cube_pose7(self) -> np.ndarray:
        return self._world_qpos[self.cube_qadr : self.cube_qadr + N_CUBE_QPOS].copy()

    def warmup_grasp(self, cube_pose7: np.ndarray, finger_ctrl: np.ndarray,
                     ref_wrist_pos: np.ndarray, ref_wrist_quat: np.ndarray,
                     n_steps: int) -> None:
        """Kinematic-pin warmup: step physics for n_steps with the cube
        qpos overwritten + qvel zeroed after every substep so fingers
        settle around a stationary cube. Pulls qpos/qvel back to host once
        at the end."""
        if n_steps <= 0:
            return
        ctrl = np.asarray(finger_ctrl, dtype=np.float64).copy()
        ctrl = np.clip(ctrl, self.finger_act_lo, self.finger_act_hi)
        self._world_ctrl[:N_FINGER_CTRL] = ctrl
        self._world_mocap_pos[self.wrist_mocap_idx] = ref_wrist_pos
        self._world_mocap_quat[self.wrist_mocap_idx] = ref_wrist_quat
        self._world_qpos[self.cube_qadr : self.cube_qadr + N_CUBE_QPOS] = cube_pose7
        self._world_qvel[self.cube_dofadr : self.cube_dofadr + 6] = 0.0
        self._push_world_state()
        cube_qadr = self.cube_qadr
        cube_dofadr = self.cube_dofadr
        for _ in range(n_steps):
            wp.capture_launch(self._graph_world)
            # Pull → re-pin cube → push back. Cheap (only 50 substeps).
            qp = self.dw_world.qpos.numpy()[0].copy()
            qv = self.dw_world.qvel.numpy()[0].copy()
            qp[cube_qadr : cube_qadr + N_CUBE_QPOS] = cube_pose7
            qv[cube_dofadr : cube_dofadr + 6] = 0.0
            self.dw_world.qpos.assign(qp[None].astype(np.float32))
            self.dw_world.qvel.assign(qv[None].astype(np.float32))
        self._world_qpos[:] = self.dw_world.qpos.numpy()[0]
        self._world_qvel[:] = self.dw_world.qvel.numpy()[0]

    @property
    def world(self):
        """Compatibility shim: code that reads env.world.qpos[:27] etc."""
        return _WorldView(self)

    # ── world stepping ───────────────────────────────────────────────────

    def step_world(self, action: np.ndarray, ref_wrist_pos: np.ndarray,
                   ref_wrist_quat: np.ndarray, n_substeps: int,
                   next_action: np.ndarray | None = None,
                   next_ref_wrist_pos: np.ndarray | None = None,
                   next_ref_wrist_quat: np.ndarray | None = None,
                   record_stride: int = 0):
        """Apply the chosen control step to the outer world (interpolating toward
        next_* across substeps when given — smooth control spline instead of a
        zero-order-hold staircase). Returns a dense (n_rec, 34) array of
        [hand_qpos(27), cube_pose7] when record_stride>0, else None. Pulls the
        resulting qpos back to host for the next MPC tick."""
        a1 = action if next_action is None else next_action
        p1 = ref_wrist_pos if next_ref_wrist_pos is None else next_ref_wrist_pos
        q1 = ref_wrist_quat if next_ref_wrist_quat is None else next_ref_wrist_quat
        rec = [] if record_stride > 0 else None
        for s in range(n_substeps):
            alpha = s / n_substeps
            a = (1.0 - alpha) * action + alpha * a1
            rp = (1.0 - alpha) * ref_wrist_pos + alpha * p1
            rq = slerp_wxyz(ref_wrist_quat, q1, alpha)
            finger_ctrl = np.clip(a[SLICE_FINGER],
                                  self.finger_act_lo, self.finger_act_hi)
            wrist_quat = quat_mul_wxyz(rq, axis_angle_to_quat_wxyz(a[SLICE_WRIST]))
            self._world_ctrl[:N_FINGER_CTRL] = finger_ctrl
            self._world_mocap_pos[self.wrist_mocap_idx] = rp
            self._world_mocap_quat[self.wrist_mocap_idx] = wrist_quat
            self.dw_world.ctrl.assign(self._world_ctrl[None].astype(np.float32))
            self.dw_world.mocap_pos.assign(self._world_mocap_pos[None].astype(np.float32))
            self.dw_world.mocap_quat.assign(self._world_mocap_quat[None].astype(np.float32))
            wp.capture_launch(self._graph_world)
            if rec is not None and (s % record_stride == 0):
                self._world_qpos[:] = self.dw_world.qpos.numpy()[0]
                rec.append(np.concatenate([self._world_qpos[:N_HAND_QPOS],
                                           self.get_world_cube_pose7()]))
        self._world_qpos[:] = self.dw_world.qpos.numpy()[0]
        self._world_qvel[:] = self.dw_world.qvel.numpy()[0]
        if rec is not None:
            rec.append(np.concatenate([self._world_qpos[:N_HAND_QPOS],
                                       self.get_world_cube_pose7()]))
            return np.asarray(rec)
        return None

    # ── batched rollout ──────────────────────────────────────────────────

    def snapshot_world_into_batch(self) -> None:
        """Broadcast world state to all N batch worlds."""
        qpos_N = np.broadcast_to(self._world_qpos[None], (self.num_samples, self.nq)).astype(np.float32)
        qvel_N = np.broadcast_to(self._world_qvel[None], (self.num_samples, self.nv)).astype(np.float32)
        mp_N = np.broadcast_to(self._world_mocap_pos[None], (self.num_samples, self.nmocap, 3)).astype(np.float32)
        mq_N = np.broadcast_to(self._world_mocap_quat[None], (self.num_samples, self.nmocap, 4)).astype(np.float32)
        ctrl_N = np.zeros((self.num_samples, self.nu), dtype=np.float32)
        self.dw_batch.qpos.assign(np.ascontiguousarray(qpos_N))
        self.dw_batch.qvel.assign(np.ascontiguousarray(qvel_N))
        self.dw_batch.mocap_pos.assign(np.ascontiguousarray(mp_N))
        self.dw_batch.mocap_quat.assign(np.ascontiguousarray(mq_N))
        self.dw_batch.ctrl.assign(ctrl_N)

    def rollout_with_resample(
        self,
        actions: np.ndarray,            # (N, H, 23)  — MAY be modified in place
        ref_wrist_pos: np.ndarray,      # (H, 3)
        ref_wrist_quat: np.ndarray,     # (H, 4) wxyz
        n_substeps: int,
        ref_cube_pos: np.ndarray,       # (H, 3)
        drop_threshold_m: float | None = None,
    ) -> dict:
        N = self.num_samples
        H = actions.shape[1]
        obs = {
            "qpos": np.zeros((N, H, self.nq), dtype=np.float64),
            "qvel": np.zeros((N, H, self.nv), dtype=np.float64),
            "cube_pos": np.zeros((N, H, 3), dtype=np.float64),
            "cube_quat_wxyz": np.zeros((N, H, 4), dtype=np.float64),
            "tip_xpos": np.zeros((N, H, 5, 3), dtype=np.float64),
            "tip_in_contact": np.zeros((N, H, 5), dtype=bool),  # unused; for API parity
        }
        terminated = np.zeros(N, dtype=bool)
        rng = np.random.default_rng()

        ctrl_buf = np.zeros((N, self.nu), dtype=np.float32)
        mocap_pos_buf = np.broadcast_to(
            self._world_mocap_pos[None], (N, self.nmocap, 3)).astype(np.float32).copy()
        mocap_quat_buf = np.broadcast_to(
            self._world_mocap_quat[None], (N, self.nmocap, 4)).astype(np.float32).copy()

        cube_qadr = self.cube_qadr

        for h in range(H):
            action_h = actions[:, h]                                # (N, 23)
            finger = np.clip(action_h[:, SLICE_FINGER],
                             self.finger_act_lo, self.finger_act_hi)
            ctrl_buf[:, :N_FINGER_CTRL] = finger

            delta = action_h[:, SLICE_WRIST]                        # (N, 3)
            delta_q = _batched_axis_angle_to_quat(delta)             # (N, 4)
            ref_q_b = np.broadcast_to(ref_wrist_quat[h][None], (N, 4))
            wrist_q = _batched_quat_mul(ref_q_b, delta_q)            # (N, 4)
            mocap_pos_buf[:, self.wrist_mocap_idx] = ref_wrist_pos[h]
            mocap_quat_buf[:, self.wrist_mocap_idx] = wrist_q

            self.dw_batch.ctrl.assign(ctrl_buf)
            self.dw_batch.mocap_pos.assign(mocap_pos_buf)
            self.dw_batch.mocap_quat.assign(mocap_quat_buf)

            for _ in range(n_substeps):
                wp.capture_launch(self._graph_batch)

            qpos_N = self.dw_batch.qpos.numpy()                      # (N, nq)
            qvel_N = self.dw_batch.qvel.numpy()                      # (N, nv)
            geom_xpos = self.dw_batch.geom_xpos.numpy()              # (N, ngeom, 3)
            obs["qpos"][:, h] = qpos_N
            obs["qvel"][:, h] = qvel_N
            cube_pos = qpos_N[:, cube_qadr : cube_qadr + 3]
            obs["cube_pos"][:, h] = cube_pos
            obs["cube_quat_wxyz"][:, h] = qpos_N[:, cube_qadr + 3 : cube_qadr + 7]
            for f, gid in enumerate(self.tip_gids):
                obs["tip_xpos"][:, h, f] = geom_xpos[:, gid]

            if (drop_threshold_m is not None
                    and h < H - 1
                    and not terminated.all()):
                err = np.linalg.norm(cube_pos - ref_cube_pos[h], axis=-1)
                dropped = (err > drop_threshold_m) & ~terminated
                good = ~dropped & ~terminated
                if dropped.any() and good.any():
                    bad_idx = np.where(dropped)[0]
                    good_idx = np.where(good)[0]
                    picks = rng.choice(good_idx, size=bad_idx.shape[0], replace=True)
                    qvel_N = self.dw_batch.qvel.numpy().copy()
                    mp_N = mocap_pos_buf.copy()
                    mq_N = mocap_quat_buf.copy()
                    ctrl_N = ctrl_buf.copy()
                    qpos_N_w = qpos_N.copy()
                    qpos_N_w[bad_idx] = qpos_N[picks]
                    qvel_N[bad_idx] = qvel_N[picks]
                    mp_N[bad_idx] = mp_N[picks]
                    mq_N[bad_idx] = mq_N[picks]
                    ctrl_N[bad_idx] = ctrl_N[picks]
                    actions[bad_idx, : h + 1] = actions[picks, : h + 1]
                    for key in ("qpos", "qvel", "cube_pos", "cube_quat_wxyz", "tip_xpos"):
                        obs[key][bad_idx, : h + 1] = obs[key][picks, : h + 1]
                    self.dw_batch.qpos.assign(qpos_N_w.astype(np.float32))
                    self.dw_batch.qvel.assign(qvel_N.astype(np.float32))
                    self.dw_batch.mocap_pos.assign(mp_N)
                    self.dw_batch.mocap_quat.assign(mq_N)
                    self.dw_batch.ctrl.assign(ctrl_N)
                    mocap_pos_buf = mp_N
                    mocap_quat_buf = mq_N
                elif dropped.all():
                    terminated |= dropped

        obs["terminated"] = terminated
        return obs


class _WorldView:
    """Read-only view of env.world that mirrors CpuEnv.world.qpos etc.
    Returns a 34-vector: [hand_qpos (27), cube_pose7 (7)].
    """
    def __init__(self, env: "MjwpEnv"):
        self._env = env

    @property
    def qpos(self):
        env = self._env
        out = np.zeros(N_HAND_QPOS + N_CUBE_QPOS, dtype=np.float64)
        out[:N_HAND_QPOS] = env._world_qpos[:N_HAND_QPOS]
        out[N_HAND_QPOS:] = env.get_world_cube_pose7()
        return out
