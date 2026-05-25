"""CPU MuJoCo environment for Stage 3 sampling MPC.

Loads the base scene (cube as a real freejoint, no virtual stick). Action
vector is 23-dim per step:

    a[ 0:20] — finger position-actuator targets (replaces data.ctrl[:20])
    a[20:23] — wrist-orientation axis-angle delta (mocap rotation perturb)

Wrist *position* is always pinned to the reference; the wrist freejoint
is anchored to the wrist mocap body by a WELD in the base XML. Cube is
free under physics — fingers must grip and rotate it without any
actuator assistance.

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

from scripts.retargeting.build_ik_scene import compile_dyn


# Action / state layout constants.
N_HAND_QPOS = 27                 # wrist freejoint (7) + 20 finger joints
N_CUBE_QPOS = 7                  # cube freejoint: xyz + wxyz quat

N_FINGER_CTRL = 20
N_WRIST_DELTA = 3
N_ACTION = N_FINGER_CTRL + N_WRIST_DELTA  # 23

SLICE_FINGER = slice(0, N_FINGER_CTRL)
SLICE_WRIST = slice(N_FINGER_CTRL, N_FINGER_CTRL + N_WRIST_DELTA)

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
TIP_GEOMS = [f"{f}_tip" for f in FINGER_NAMES]
TIP_BODIES = ["rl_dg_1_4", "rl_dg_2_4", "rl_dg_3_4", "rl_dg_4_4", "rl_dg_5_4"]
CUBE_GEOM = "tracked_cube_geom"
WRIST_MOCAP = "wrist_mocap"
CUBE_FREEJOINT = "cube_freejoint"

# Default sim timestep. SPIDER-matched: 0.01 (their setup_mj_model). Was 0.002.
DEFAULT_SIM_DT = 0.01

# SPIDER contact model (spider/simulators/mjwp.py setup_mj_model + generate_xml):
#   - soft contacts ("softer contact for sim2real"): solimp dmin=0, width=0.03
#   - condim 4 on the cube (torsional friction active → grip resists cube spin;
#     ours was 3, which let the cube pivot freely on a fingertip)
#   - object friction [1, 1, 0.1] (torsional 1.0 active at condim 4)
# These are the levers SPIDER uses *instead of* domain randomization.
SPIDER_SOLIMP = [0.0, 0.95, 0.03, 0.5, 2.0]
SPIDER_FRICTION = [1.0, 1.0, 0.1]
SPIDER_CUBE_CONDIM = 4


def apply_spider_physics(model) -> None:
    """Mutate `model` in place to SPIDER's soft-contact + torsional-friction
    setup on the cube and fingertip geoms. See SPIDER_* constants above."""
    cube_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
    tip_gids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in TIP_GEOMS]
    # condim of a contact is max(geom1, geom2), so setting the cube to 4 makes
    # every cube↔finger contact condim 4.
    model.geom_condim[cube_gid] = SPIDER_CUBE_CONDIM
    for gid in [cube_gid, *tip_gids]:
        model.geom_solimp[gid] = SPIDER_SOLIMP
        model.geom_friction[gid] = SPIDER_FRICTION


@dataclass
class EnvIds:
    wrist_mocap: int
    cube_qadr: int               # qpos start of the 7-DOF cube freejoint
    cube_dofadr: int             # qvel start of the 6-DOF cube velocity
    cube_geom_id: int
    tip_geom_ids: list[int]
    finger_act_lo: np.ndarray    # (20,)  ctrlrange[:, 0] for fingers
    finger_act_hi: np.ndarray    # (20,)


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


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation between wxyz quaternions (alpha in [0,1])."""
    q0 = np.asarray(q0, dtype=float); q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1; d = -d
    if d > 0.9995:                      # nearly parallel → normalized lerp
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(d)
    s = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / s


class CpuEnv:
    """Sequential CPU rollout env. Owns 1 world + N batch MjData."""

    def __init__(self, num_samples: int, sim_dt: float = DEFAULT_SIM_DT,
                 spider_physics: bool = True):
        self.model, _ = compile_dyn(sim_dt=sim_dt)
        # Match MjwpEnv's solver config so a cpu rollout compares apples-to-
        # apples with an mjwp-planned one. Findings (2026-05): integrator and
        # iteration count are no-ops for this model (implicit-fast ≡ Euler with
        # no passive damping; the solver converges well under 20 iters), but
        # impratio IS real — mujoco_warp does NOT honor impratio (its Option
        # has no such field; it behaves as impratio=1), while the XML asks for
        # 10. Leaving cpu at 10 ran *stiffer friction* than the GPU planner.
        # Matching to 1 isolates the residual gap to float32-vs-float64 +
        # warp's separate solver implementation (which is not configurable).
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.iterations = 20
        self.model.opt.ls_iterations = 50
        self.model.opt.impratio = 1.0
        if spider_physics:
            apply_spider_physics(self.model)
        self.num_samples = num_samples
        self.world = mujoco.MjData(self.model)
        self.batch = [mujoco.MjData(self.model) for _ in range(num_samples)]
        self.ids = self._resolve_ids()
        if self.model.nq != N_HAND_QPOS + N_CUBE_QPOS:
            raise ValueError(f"Model nq={self.model.nq}, expected "
                             f"{N_HAND_QPOS + N_CUBE_QPOS}")
        if self.model.nu != N_FINGER_CTRL:
            raise ValueError(f"Model nu={self.model.nu}, expected "
                             f"{N_FINGER_CTRL}")

    def _resolve_ids(self) -> EnvIds:
        m = self.model
        wrist_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, WRIST_MOCAP)
        wrist_mid = int(m.body_mocapid[wrist_bid])
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, CUBE_FREEJOINT)
        cube_qadr = int(m.jnt_qposadr[cube_jid])
        cube_dofadr = int(m.jnt_dofadr[cube_jid])
        cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
        tip_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in TIP_GEOMS]
        if any(i < 0 for i in (wrist_mid, cube_qadr, cube_dofadr, cube_gid, *tip_gids)):
            raise ValueError("Failed to resolve required ids in dyn scene")
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

    # ── world I/O ─────────────────────────────────────────────────────────

    def set_world_state(self, hand_qpos: np.ndarray, cube_pose7: np.ndarray,
                        qvel: np.ndarray | None = None) -> None:
        self.world.qpos[:N_HAND_QPOS] = hand_qpos
        self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + N_CUBE_QPOS] = cube_pose7
        self.world.qvel[:] = 0.0 if qvel is None else qvel
        self.world.mocap_pos[self.ids.wrist_mocap] = hand_qpos[0:3]
        self.world.mocap_quat[self.ids.wrist_mocap] = hand_qpos[3:7]
        mujoco.mj_forward(self.model, self.world)

    def get_world_cube_pose7(self) -> np.ndarray:
        cube_q = self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + N_CUBE_QPOS]
        return cube_q.copy()

    def get_world_cube_pos(self) -> np.ndarray:
        return self.world.qpos[self.ids.cube_qadr : self.ids.cube_qadr + 3].copy()

    def warmup_grasp(self, cube_pose7: np.ndarray, finger_ctrl: np.ndarray,
                     ref_wrist_pos: np.ndarray, ref_wrist_quat: np.ndarray,
                     n_steps: int) -> None:
        """SPIDER-style virtual-stick initial grasp via *kinematic pin*: the
        cube qpos is overwritten and its qvel zeroed after every substep so
        fingers settle around a stationary cube. After warmup the cube is
        released and physics takes over."""
        if n_steps <= 0:
            return
        cube_qadr = self.ids.cube_qadr
        cube_dofadr = self.ids.cube_dofadr
        ctrl = np.asarray(finger_ctrl, dtype=np.float64).copy()
        ctrl = np.clip(ctrl, self.ids.finger_act_lo, self.ids.finger_act_hi)
        self.world.qpos[cube_qadr : cube_qadr + N_CUBE_QPOS] = cube_pose7
        self.world.qvel[cube_dofadr : cube_dofadr + 6] = 0.0
        self.world.ctrl[:] = ctrl
        self.world.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos
        self.world.mocap_quat[self.ids.wrist_mocap] = ref_wrist_quat
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.world)
            self.world.qpos[cube_qadr : cube_qadr + N_CUBE_QPOS] = cube_pose7
            self.world.qvel[cube_dofadr : cube_dofadr + 6] = 0.0

    # ── action application ────────────────────────────────────────────────

    def _apply_action_to_data(self, data: mujoco.MjData, action: np.ndarray,
                              ref_wrist_pos: np.ndarray,
                              ref_wrist_quat: np.ndarray) -> None:
        finger_ctrl = np.clip(action[SLICE_FINGER],
                              self.ids.finger_act_lo, self.ids.finger_act_hi)
        delta = action[SLICE_WRIST]
        wrist_quat = quat_mul_wxyz(ref_wrist_quat, axis_angle_to_quat_wxyz(delta))
        data.ctrl[:N_FINGER_CTRL] = finger_ctrl
        data.mocap_pos[self.ids.wrist_mocap] = ref_wrist_pos
        data.mocap_quat[self.ids.wrist_mocap] = wrist_quat

    def step_world(self, action: np.ndarray, ref_wrist_pos: np.ndarray,
                   ref_wrist_quat: np.ndarray, n_substeps: int,
                   next_action: np.ndarray | None = None,
                   next_ref_wrist_pos: np.ndarray | None = None,
                   next_ref_wrist_quat: np.ndarray | None = None,
                   record_stride: int = 0):
        """Advance the world one control step.

        If `next_*` are given, the applied control + reference wrist are linearly
        interpolated (SLERP for the wrist quat) from this control step toward the
        next across the n_substeps — a smooth control spline instead of a
        0.4 s zero-order-hold staircase. Returns a dense (n_rec, 34) array of
        [hand_qpos(27), cube_pose7] when record_stride>0, else None."""
        a1 = action if next_action is None else next_action
        p1 = ref_wrist_pos if next_ref_wrist_pos is None else next_ref_wrist_pos
        q1 = ref_wrist_quat if next_ref_wrist_quat is None else next_ref_wrist_quat
        rec = [] if record_stride > 0 else None
        for s in range(n_substeps):
            alpha = s / n_substeps
            a = (1.0 - alpha) * action + alpha * a1
            rp = (1.0 - alpha) * ref_wrist_pos + alpha * p1
            rq = slerp_wxyz(ref_wrist_quat, q1, alpha)
            self._apply_action_to_data(self.world, a, rp, rq)
            mujoco.mj_step(self.model, self.world)
            if rec is not None and (s % record_stride == 0):
                rec.append(self.world.qpos[:N_HAND_QPOS + N_CUBE_QPOS].copy())
        if rec is not None:
            rec.append(self.world.qpos[:N_HAND_QPOS + N_CUBE_QPOS].copy())
            return np.asarray(rec)
        return None

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
        actions: np.ndarray,            # (N, H, 23)  — MAY be modified in place
        ref_wrist_pos: np.ndarray,      # (H, 3)
        ref_wrist_quat: np.ndarray,     # (H, 4)
        n_substeps: int,
        ref_cube_pos: np.ndarray,       # (H, 3)
        drop_threshold_m: float | None = None,
    ) -> dict:
        """Roll all N samples through the H-step horizon, stepping
        horizon-by-horizon so we can swap dropped-cube samples with good
        ones between steps (SPIDER resample-on-drop). Returns observations
        used by the reward function plus per-sample terminate flag.
        """
        N = self.num_samples
        H = actions.shape[1]
        obs = {
            "qpos": np.zeros((N, H, self.model.nq), dtype=np.float64),
            "qvel": np.zeros((N, H, self.model.nv), dtype=np.float64),
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
                obs["qpos"][i, h] = d.qpos
                obs["qvel"][i, h] = d.qvel
                obs["cube_pos"][i, h] = d.qpos[cube_qadr : cube_qadr + 3]
                obs["cube_quat_wxyz"][i, h] = d.qpos[cube_qadr + 3 : cube_qadr + 7]
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
                        actions[bi, : h + 1] = actions[gi, : h + 1]
                        for key in obs:
                            obs[key][bi, : h + 1] = obs[key][gi, : h + 1]
                elif dropped.all():
                    terminated |= dropped

        obs["terminated"] = terminated
        return obs
