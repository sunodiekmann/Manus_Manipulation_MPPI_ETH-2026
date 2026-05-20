"""Build IK and dynamics MuJoCo scenes from the Tesollo base XML.

`assets/output_mocap_physics.xml` is the single source of truth. This module
augments it programmatically with:

  * `build_ik_spec(...)`: sites on fingertips/palm/cube, mocap target bodies,
    and equality springs (CONNECT for fingertips, WELD for palm and cube)
    that pull the hand toward the targets. Mirrors SPIDER's IK setup.

  * `compile_dyn(...)`: returns the base scene unchanged. The cube keeps its
    natural freejoint — no replacement joints, no actuators, no virtual
    stick. Stage 3 MPC drives only the fingers + a wrist-orientation delta
    and lets physics handle the cube freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


BASE_XML = Path(__file__).resolve().parents[2] / "assets" / "output_mocap_physics.xml"

WRIST_MOCAP_BODY = "wrist_mocap"
CUBE_BODY = "tracked_cube"
CUBE_TARGET_MOCAP_BODY = "cube_target"


@dataclass(frozen=True)
class FingerTip:
    finger: str            # mapping key used by tesollo_constants.FINGERS
    body: str              # distal-phalanx body in the URDF
    target_body: str       # mocap body that the IK target snaps to
    local_offset: tuple    # site position in `body` frame (matches the *_tip geom)
    width_scale: float     # SPIDER-style solimp width multiplier


FINGER_TIPS: tuple[FingerTip, ...] = (
    FingerTip("thumb_tip",  "rl_dg_1_4", "thumb_tip_target",  (0.0, 0.0363, 0.0),  1.0),
    FingerTip("index_tip",  "rl_dg_2_4", "index_tip_target",  (0.0, 0.0,    0.0255), 1.0),
    FingerTip("middle_tip", "rl_dg_3_4", "middle_tip_target", (0.0, 0.0,    0.0255), 1.0),
    FingerTip("ring_tip",   "rl_dg_4_4", "ring_tip_target",   (0.0, 0.0,    0.0255), 3.0),
    FingerTip("little_tip", "rl_dg_5_4", "little_tip_target", (0.0, 0.0,    0.0363), 3.0),
)


# Equality solver parameters.
#   solref[0] = time constant. ~2x sim_dt — stable integration but closes
#   ≈85% of the gap per ref frame so the fingers don't visibly lag the targets.
#   solimp d0 > 0 — without this, the spring goes near-zero inside `width` of
#   the target, so fingers loiter in a 10mm deadband.
FINGER_SOLREF = (0.02, 1.0)
FINGER_SOLIMP_BASE = (0.6, 0.95, 0.01, 0.5, 2.0)  # width (idx 2) scaled per-finger

CUBE_SOLREF = (0.002, 1.0)
CUBE_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
CUBE_TORQUE_SCALE = 10.0

# Solver iteration counts. SPIDER bumps these for the dense-equality IK scene.
IK_SOLVER_ITERATIONS = 20
IK_SOLVER_LS_ITERATIONS = 50


def _add_site(body, name, pos, size=(0.005,) * 3, rgba=(0.0, 1.0, 0.0, 0.5)):
    return body.add_site(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=list(pos),
        size=list(size),
        rgba=list(rgba),
        group=4,
    )


def _add_equality(spec, name, site_a, site_b, eq_type, solref, solimp, torque_scale=1.0):
    data = np.zeros(11)
    data[10] = torque_scale
    eq = spec.add_equality(
        name=name,
        type=eq_type,
        name1=site_a,
        name2=site_b,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        data=data,
    )
    eq.solref = list(solref)
    eq.solimp = list(solimp)
    return eq


def _add_target_site_on_existing_mocap(spec, body_name, site_name):
    """The base XML already has *_target mocap bodies and wrist_mocap; we
    just need to attach a named site to each so equality constraints can
    reference them by site name (SPIDER convention)."""
    body = spec.body(body_name)
    return _add_site(body, site_name, pos=(0.0, 0.0, 0.0), rgba=(1.0, 0.5, 0.0, 0.5))


def build_ik_spec(*, base_xml: Path | str = BASE_XML) -> mujoco.MjSpec:
    """Augment the base scene with mocap targets + equality springs for IK.

    The returned spec is *not* compiled — call `.compile()` after any further
    tweaks. Collisions and actuation are disabled in the model `opt` flags so
    the IK is purely driven by the springs.
    """
    spec = mujoco.MjSpec.from_file(str(base_xml))

    # Sites on the robot side (fingertips, cube).
    for ft in FINGER_TIPS:
        body = spec.body(ft.body)
        _add_site(body, f"{ft.finger}_site", pos=ft.local_offset)
    cube_body = spec.body(CUBE_BODY)
    _add_site(cube_body, "cube_site", pos=(0.0, 0.0, 0.0))

    # Add a cube_target mocap body (fingertip + wrist targets already exist).
    cube_target = spec.worldbody.add_body(
        name=CUBE_TARGET_MOCAP_BODY,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        mocap=True,
    )
    _add_site(cube_target, f"{CUBE_TARGET_MOCAP_BODY}_site", pos=(0.0, 0.0, 0.0),
              rgba=(1.0, 0.5, 0.0, 0.5))

    # Sites on existing fingertip target mocap bodies (so equalities can
    # reference them by site name).
    for ft in FINGER_TIPS:
        _add_target_site_on_existing_mocap(spec, ft.target_body, f"{ft.target_body}_site")

    # Base XML has body-level WELD (rh ↔ wrist_mocap). The default solver
    # params are too soft against five fingertip CONNECT springs pulling on
    # the kinematic chain, so we upgrade to cube-style stiffness here.
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_WELD:
            eq.solref = list(CUBE_SOLREF)
            eq.solimp = list(CUBE_SOLIMP)
            data = np.array(eq.data, dtype=float).copy()
            data[10] = CUBE_TORQUE_SCALE
            eq.data = data
            break

    # Fingertip CONNECT (position-only) equalities.
    for ft in FINGER_TIPS:
        solimp = list(FINGER_SOLIMP_BASE)
        solimp[2] = FINGER_SOLIMP_BASE[2] * ft.width_scale
        _add_equality(
            spec,
            name=f"{ft.finger}_eq",
            site_a=f"{ft.finger}_site",
            site_b=f"{ft.target_body}_site",
            eq_type=mujoco.mjtEq.mjEQ_CONNECT,
            solref=FINGER_SOLREF,
            solimp=tuple(solimp),
            torque_scale=1.0,
        )

    # Wrist WELD lives in the base XML (body1=rh, body2=wrist_mocap) — kept
    # as-is.

    # Cube WELD (pose).
    _add_equality(
        spec,
        name="cube_eq",
        site_a="cube_site",
        site_b=f"{CUBE_TARGET_MOCAP_BODY}_site",
        eq_type=mujoco.mjtEq.mjEQ_WELD,
        solref=CUBE_SOLREF,
        solimp=CUBE_SOLIMP,
        torque_scale=CUBE_TORQUE_SCALE,
    )

    return spec


def compile_ik(*, base_xml: Path | str = BASE_XML, sim_dt: float = 0.01,
               disable_collisions: bool = True, dump_to: Path | str | None = None):
    """Compile the IK spec and apply the solver/disable flags SPIDER uses."""
    spec = build_ik_spec(base_xml=base_xml)
    if dump_to is not None:
        Path(dump_to).write_text(spec.to_xml())
    model = spec.compile()
    model.opt.timestep = sim_dt
    model.opt.iterations = IK_SOLVER_ITERATIONS
    model.opt.ls_iterations = IK_SOLVER_LS_ITERATIONS
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    if disable_collisions:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    # IK is kinematic; gravity and body inertia would fight the equality
    # springs. Zero gravity and shrink masses to make the equality solver
    # dominate. Cube mass is left alone (the WELD already pins it stiffly,
    # and Stage 3 reads back the same cube body so we don't want to change
    # its inertia here).
    model.opt.gravity[:] = 0.0
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    for bid in range(model.nbody):
        if bid == cube_body_id:
            continue
        model.body_mass[bid] *= 1e-3
        model.body_inertia[bid] *= 1e-3
    return model, mujoco.MjData(model)


def compile_dyn(*, base_xml: Path | str = BASE_XML, sim_dt: float = 0.01,
                dump_to: Path | str | None = None):
    """Compile the base scene as-is for Stage 3 MPC.

    The cube keeps its natural freejoint — no joint replacement, no
    actuators, no virtual stick. Stage 3 drives only the hand (20 finger
    actuators + a wrist-mocap orientation perturbation) and lets the
    cube move freely under contact physics.
    """
    spec = mujoco.MjSpec.from_file(str(base_xml))
    if dump_to is not None:
        Path(dump_to).write_text(spec.to_xml())
    model = spec.compile()
    model.opt.timestep = sim_dt
    return model, mujoco.MjData(model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build IK + dyn MuJoCo scenes.")
    parser.add_argument("--dump", action="store_true",
                        help="Write the compiled XMLs to assets/output_ik_physics.xml "
                             "and assets/output_dyn_physics.xml for inspection.")
    args = parser.parse_args()

    assets_dir = Path(__file__).resolve().parents[2] / "assets"
    ik_dump = assets_dir / "output_ik_physics.xml" if args.dump else None
    dyn_dump = assets_dir / "output_dyn_physics.xml" if args.dump else None

    ik_model, _ = compile_ik(dump_to=ik_dump)
    dyn_model, _ = compile_dyn(dump_to=dyn_dump)

    print(f"IK : nq={ik_model.nq}, nv={ik_model.nv}, neq={ik_model.neq}, "
          f"nsite={ik_model.nsite}, mocaps={ik_model.nmocap}")
    print(f"DYN: nq={dyn_model.nq}, nv={dyn_model.nv}, nu={dyn_model.nu}")
    if args.dump:
        print(f"Wrote {ik_dump}\nWrote {dyn_dump}")
