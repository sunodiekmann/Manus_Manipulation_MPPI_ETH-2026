"""GPU-batched rollout for Stage 3 sampling MPC, on MuJoCo Warp.

Mirrors `cpu_env.CpuEnv` semantics — same dyn scene, same 29-dim action
layout, same `rollout_with_resample` signature — but with N worlds stepped
in parallel via NVIDIA Warp CUDA kernels. SPIDER uses this backend for the
same reason: a single GPU dispatch of N=2048 × H=8 horizon steps × n
substeps is orders of magnitude faster than sequential CPU rollouts.

Host↔device data shuffling per MPC tick is small (action knots in, cube
state out for resample-on-drop, reward computed host-side from arrays the
size of N samples). The first call compiles + caches Warp kernels on disk
(~30s); subsequent runs are fast.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjwp
import numpy as np
import warp as wp

from scripts.retargeting.build_ik_scene import compile_dyn, scale_obj_actuators
from scripts.retargeting.dynamic_retargeting.cpu_env import (
    CUBE_GEOM,
    N_ACTION,
    N_CUBE_QPOS,
    N_FINGER_CTRL,
    N_HAND_QPOS,
    N_OBJ_CTRL,
    N_OBJ_POS_CTRL,
    N_WRIST_DELTA,
    SLICE_FINGER,
    SLICE_OBJ_POS,
    SLICE_OBJ_ROT,
    SLICE_WRIST,
    TIP_GEOMS,
    WRIST_MOCAP,
    axis_angle_to_quat_wxyz,
    cube_qpos_from_pose7,
    cube_pose7_from_qpos,
    euler_xyz_to_quat_wxyz,
    quat_mul_wxyz,
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
    snapshot_world_into_batch, rollout_with_resample, set_obj_decay,
    set_closure_bias, get_world_cube_*), but stepping N worlds in parallel
    on GPU via mujoco_warp.
    """

    def __init__(self, num_samples: int, sim_dt: float = 0.002):
        self.mj_model, _ = compile_dyn(sim_dt=sim_dt)
        # SPIDER's mjwp settings: implicitfast integrator, moderate solver iters.
        self.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.mj_model.opt.iterations = 20
        self.mj_model.opt.ls_iterations = 50
        if self.mj_model.nu != N_FINGER_CTRL + N_OBJ_CTRL:
            raise ValueError(f"Model nu={self.mj_model.nu}, expected "
                             f"{N_FINGER_CTRL + N_OBJ_CTRL}")
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
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_pos_x")
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

        # Two parallel Data objects: one with nworld=1 for the outer world,
        # and one with nworld=N for the rollout batch. Both share the same
        # Model. We cache the max-gain copies of actuator_gainprm/biasprm
        # so set_obj_decay can write scaled values in-place without
        # invalidating the CUDA Graph capture below.
        self.mw = mjwp.put_model(m)
        self._max_gainprm = self.mw.actuator_gainprm.numpy().copy()
        self._max_biasprm = self.mw.actuator_biasprm.numpy().copy()
        # Preallocate constraint workspace generously. mjwp warns
        # `nefc overflow - please increase njmax to N` and falls back to a
        # slow path when contacts spike; ~128 covers our cube + finger
        # contact set with headroom.
        self.dw_world = mjwp.make_data(m, nworld=1, njmax=256)
        self.dw_batch = mjwp.make_data(m, nworld=self.num_samples, njmax=256)

        # CUDA Graph capture for both steppers — replaces the ~4 ms launch
        # overhead per mjwp.step with a single graph dispatch (~50 us at
        # any N). Warmup once so kernels compile, then capture.
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
        # Host-side mirrors of the world state. Kept so we don't have to
        # round-trip qpos every time we need to read it.
        self._world_qpos = np.zeros(self.nq, dtype=np.float64)
        self._world_qvel = np.zeros(self.nv, dtype=np.float64)
        self._world_mocap_pos = np.zeros((self.nmocap, 3), dtype=np.float64)
        self._world_mocap_quat = np.zeros((self.nmocap, 4), dtype=np.float64)
        self._world_mocap_quat[:, 0] = 1.0
        self._world_ctrl = np.zeros(self.nu, dtype=np.float64)

        self.ctrl_bias = np.zeros(N_FINGER_CTRL, dtype=np.float64)
        self.obj_decay = 1.0

    # ── ergonomic API mirror of CpuEnv ────────────────────────────────────

    def set_obj_decay(self, factor: float) -> None:
        """Scale the 6 cube actuator gains in-place. The CUDA Graph captured
        in __init__ closes over the device pointers of mw.actuator_gainprm /
        biasprm — writing into those arrays takes effect on subsequent
        graph launches without recapture.
        """
        self.obj_decay = float(factor)
        # We don't bother with the mj_model copy here — that path is only
        # used to compile the dyn spec once. Apply the decay directly to
        # the device arrays via the cached max values.
        scaled_gain = self._max_gainprm.copy()
        scaled_bias = self._max_biasprm.copy()
        # Cube actuators are the last 6 (indices 20..26). gainprm/biasprm
        # have shape (1, 26, 10).
        cube_slice = slice(N_FINGER_CTRL, N_FINGER_CTRL + N_OBJ_CTRL)
        scaled_gain[:, cube_slice, 0] *= factor                       # kp
        scaled_bias[:, cube_slice, 1] *= factor                       # -kp
        scaled_bias[:, cube_slice, 2] *= factor                       # -kv
        self.mw.actuator_gainprm.assign(scaled_gain)
        self.mw.actuator_biasprm.assign(scaled_bias)

    def set_closure_bias(self, bias_rad: float) -> None:
        self.ctrl_bias = bias_rad * self._flexion_closure_direction()

    def _flexion_closure_direction(self) -> np.ndarray:
        d = np.zeros(N_FINGER_CTRL, dtype=np.float64)
        for i in range(N_FINGER_CTRL):
            lo, hi = self.mj_model.jnt_range[i + 1]
            if lo >= 0.0 and hi > 0.0:
                d[i] = 1.0
            elif hi <= 0.0 and lo < 0.0:
                d[i] = -1.0
        return d

    def set_world_state(self, hand_qpos: np.ndarray, cube_pose7: np.ndarray,
                        qvel: np.ndarray | None = None) -> None:
        cube_q = cube_qpos_from_pose7(cube_pose7)
        self._world_qpos[:N_HAND_QPOS] = hand_qpos
        self._world_qpos[self.cube_qadr : self.cube_qadr + N_CUBE_QPOS] = cube_q
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
        cube_q = self._world_qpos[self.cube_qadr : self.cube_qadr + N_CUBE_QPOS]
        return cube_pose7_from_qpos(cube_q)

    @property
    def world(self):
        """Compatibility shim: code that reads env.world.qpos[:27] etc."""
        return _WorldView(self)

    # ── world stepping ───────────────────────────────────────────────────

    def step_world(self, action: np.ndarray, ref_wrist_pos: np.ndarray,
                   ref_wrist_quat: np.ndarray, n_substeps: int) -> None:
        """Apply the chosen plan[0] to the outer world. Pulls the resulting
        qpos back to host for the next MPC tick's reference."""
        finger_ctrl = np.clip(action[SLICE_FINGER] + self.ctrl_bias,
                              self.finger_act_lo, self.finger_act_hi)
        delta = action[SLICE_WRIST]
        wrist_quat = quat_mul_wxyz(ref_wrist_quat, axis_angle_to_quat_wxyz(delta))
        self._world_ctrl[:N_FINGER_CTRL] = finger_ctrl
        self._world_ctrl[N_FINGER_CTRL : N_FINGER_CTRL + N_OBJ_POS_CTRL] = action[SLICE_OBJ_POS]
        self._world_ctrl[N_FINGER_CTRL + N_OBJ_POS_CTRL : N_FINGER_CTRL + N_OBJ_CTRL] = action[SLICE_OBJ_ROT]
        self._world_mocap_pos[self.wrist_mocap_idx] = ref_wrist_pos
        self._world_mocap_quat[self.wrist_mocap_idx] = wrist_quat

        self.dw_world.ctrl.assign(self._world_ctrl[None].astype(np.float32))
        self.dw_world.mocap_pos.assign(self._world_mocap_pos[None].astype(np.float32))
        self.dw_world.mocap_quat.assign(self._world_mocap_quat[None].astype(np.float32))
        for _ in range(n_substeps):
            wp.capture_launch(self._graph_world)
        self._world_qpos[:] = self.dw_world.qpos.numpy()[0]
        self._world_qvel[:] = self.dw_world.qvel.numpy()[0]

    # ── batched rollout ──────────────────────────────────────────────────

    def snapshot_world_into_batch(self) -> None:
        """Broadcast world state to all N batch worlds."""
        qpos_N = np.broadcast_to(self._world_qpos[None], (self.num_samples, self.nq)).astype(np.float32)
        qvel_N = np.broadcast_to(self._world_qvel[None], (self.num_samples, self.nv)).astype(np.float32)
        mp_N = np.broadcast_to(self._world_mocap_pos[None], (self.num_samples, self.nmocap, 3)).astype(np.float32)
        mq_N = np.broadcast_to(self._world_mocap_quat[None], (self.num_samples, self.nmocap, 4)).astype(np.float32)
        ctrl_N = np.zeros((self.num_samples, self.nu), dtype=np.float32)
        # `assign` requires a writable contiguous array.
        self.dw_batch.qpos.assign(np.ascontiguousarray(qpos_N))
        self.dw_batch.qvel.assign(np.ascontiguousarray(qvel_N))
        self.dw_batch.mocap_pos.assign(np.ascontiguousarray(mp_N))
        self.dw_batch.mocap_quat.assign(np.ascontiguousarray(mq_N))
        self.dw_batch.ctrl.assign(ctrl_N)

    def rollout_with_resample(
        self,
        actions: np.ndarray,            # (N, H, 29)  — MAY be modified in place
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
            "tip_in_contact": np.zeros((N, H, 5), dtype=bool),  # unused; here for API parity
        }
        terminated = np.zeros(N, dtype=bool)
        rng = np.random.default_rng()

        # Host buffers for ctrl + mocap_quat written each step.
        ctrl_buf = np.zeros((N, self.nu), dtype=np.float32)
        mocap_pos_buf = np.broadcast_to(
            self._world_mocap_pos[None], (N, self.nmocap, 3)).astype(np.float32).copy()
        mocap_quat_buf = np.broadcast_to(
            self._world_mocap_quat[None], (N, self.nmocap, 4)).astype(np.float32).copy()

        cube_qadr = self.cube_qadr

        for h in range(H):
            action_h = actions[:, h]                                # (N, 29)
            finger = np.clip(action_h[:, SLICE_FINGER] + self.ctrl_bias,
                             self.finger_act_lo, self.finger_act_hi)
            ctrl_buf[:, :N_FINGER_CTRL] = finger
            ctrl_buf[:, N_FINGER_CTRL : N_FINGER_CTRL + N_OBJ_POS_CTRL] = action_h[:, SLICE_OBJ_POS]
            ctrl_buf[:, N_FINGER_CTRL + N_OBJ_POS_CTRL : N_FINGER_CTRL + N_OBJ_CTRL] = action_h[:, SLICE_OBJ_ROT]

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

            # Pull state out for observation + resample decisions.
            qpos_N = self.dw_batch.qpos.numpy()                      # (N, nq)
            qvel_N = self.dw_batch.qvel.numpy()                      # (N, nv)
            geom_xpos = self.dw_batch.geom_xpos.numpy()              # (N, ngeom, 3)
            obs["qpos"][:, h] = qpos_N
            obs["qvel"][:, h] = qvel_N
            cube_pos = qpos_N[:, cube_qadr : cube_qadr + 3]
            obs["cube_pos"][:, h] = cube_pos
            cube_euler = qpos_N[:, cube_qadr + 3 : cube_qadr + 6]
            # Euler XYZ → wxyz quaternion (vectorised).
            cube_quat = _euler_xyz_to_quat_wxyz_batched(cube_euler)
            obs["cube_quat_wxyz"][:, h] = cube_quat
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
                    # Swap state on host arrays, then push back.
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
                    # And mirror the obs trace so reward sees a consistent history.
                    for key in ("qpos", "cube_pos", "cube_quat_wxyz", "tip_xpos"):
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


def _euler_xyz_to_quat_wxyz_batched(euler: np.ndarray) -> np.ndarray:
    """Vectorised over leading dims; matches `euler_xyz_to_quat_wxyz`."""
    cx, cy, cz = np.cos(0.5 * euler[..., 0]), np.cos(0.5 * euler[..., 1]), np.cos(0.5 * euler[..., 2])
    sx, sy, sz = np.sin(0.5 * euler[..., 0]), np.sin(0.5 * euler[..., 1]), np.sin(0.5 * euler[..., 2])
    return np.stack([
        cx * cy * cz - sx * sy * sz,
        sx * cy * cz + cx * sy * sz,
        cx * sy * cz - sx * cy * sz,
        cx * cy * sz + sx * sy * cz,
    ], axis=-1)


class _WorldView:
    """Read-only view of env.world that mirrors CpuEnv.world.qpos etc.

    optimize_sequence_dynamic.py uses `env.world.qpos[:27]` and `[27:34]`
    (7-d cube freejoint) but we have 6-d cube here. The output saving path
    expects pose7 for cube, so we reconstruct via cube_pose7_from_qpos.
    """
    def __init__(self, env: "MjwpEnv"):
        self._env = env

    @property
    def qpos(self):
        """Return a (34,) array: [hand_qpos (27), cube_pose7 (7)] — same
        layout that `optimize_sequence_dynamic.py` expects when slicing."""
        env = self._env
        cube_pose7 = env.get_world_cube_pose7()
        out = np.zeros(N_HAND_QPOS + 7, dtype=np.float64)
        out[:N_HAND_QPOS] = env._world_qpos[:N_HAND_QPOS]
        out[N_HAND_QPOS:] = cube_pose7
        return out
