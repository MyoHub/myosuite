# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Derive mjlab configuration from the CPU (Gymnasium) registration of an env id.

The CPU registration (``gymnasium.spec(env_id)``) is the single source of truth
for every task parameter (model, targets, thresholds, reward weights, frame skip,
episode length, muscle condition). mjlab task configs read it through these
helpers instead of restating numbers, so the two backends cannot drift.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sim import MujocoCfg

from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg

_INTEGRATORS = {
    int(mujoco.mjtIntegrator.mjINT_EULER): "euler",
    int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
}
_SOLVERS = {
    int(mujoco.mjtSolver.mjSOL_PGS): "pgs",
    int(mujoco.mjtSolver.mjSOL_CG): "cg",
    int(mujoco.mjtSolver.mjSOL_NEWTON): "newton",
}
_CONES = {
    int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal",
    int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic",
}
_JACOBIANS = {
    int(mujoco.mjtJacobian.mjJAC_DENSE): "dense",
    int(mujoco.mjtJacobian.mjJAC_SPARSE): "sparse",
    int(mujoco.mjtJacobian.mjJAC_AUTO): "auto",
}


@dataclass(frozen=True)
class CpuTaskSpec:
    """The parts of a CPU registration an mjlab port needs.

    Attributes:
        env_id: Gymnasium env id shared by both backends.
        kwargs: Constructor kwargs of the CPU env (defaults not included).
        max_episode_steps: CPU ``TimeLimit`` horizon.
    """

    env_id: str
    kwargs: dict[str, Any]
    max_episode_steps: int

    @property
    def muscle_condition(self) -> str:
        """``""``, ``"sarcopenia"``, ``"fatigue"`` or ``"reafferentation"``."""
        return str(self.kwargs.get("muscle_condition", ""))

    @property
    def frame_skip(self) -> int:
        """Physics substeps per control step (CPU default 10)."""
        return int(self.kwargs.get("frame_skip", 10))

    @property
    def uses_recipe(self) -> bool:
        """Whether the model comes from a ``ModelBuilder`` recipe (``_r`` names)."""
        return self.kwargs.get("model_recipe") is not None


def cpu_task_spec(env_id: str) -> CpuTaskSpec:
    """Read the CPU registration of *env_id*.

    Args:
        env_id: A registered MyoSuite CPU env id.

    Returns:
        The frozen registration data.
    """
    import myosuite  # noqa: F401, PLC0415  (registers the CPU envs)

    spec = gym.spec(env_id)
    return CpuTaskSpec(
        env_id=env_id,
        kwargs=dict(spec.kwargs),
        max_episode_steps=int(spec.max_episode_steps),
    )


def build_cpu_spec(task: CpuTaskSpec) -> mujoco.MjSpec:
    """Build the same ``MjSpec`` the CPU env compiles, plus its muscle condition.

    Args:
        task: CPU registration of the task.

    Returns:
        A fresh spec (safe to attach into an mjlab scene).
    """
    kwargs = task.kwargs
    edit_fn = kwargs.get("edit_fn")
    if task.uses_recipe:
        _, spec = build_from_recipe(kwargs["model_recipe"])
        if edit_fn is not None:
            edit_fn(spec)
    else:
        builder = ModelBuilder.from_xml_file(kwargs["model_path"])
        if edit_fn is not None:

            def _wrap(spec: mujoco.MjSpec) -> mujoco.MjSpec:
                edit_fn(spec)
                return spec

            builder = builder.apply_transform(_wrap)
        _, spec = builder.build()
    if task.muscle_condition == "sarcopenia":
        apply_sarcopenia_to_spec(spec, force_scale=0.5)
    return spec


@dataclass(frozen=True)
class CompiledModelInfo:
    """Static model data needed at config time (no ``id(env)`` state)."""

    opt_timestep: float
    mujoco_cfg: MujocoCfg
    joint_names: tuple[str, ...]
    jnt_type: tuple[int, ...]
    jnt_qposadr: tuple[int, ...]
    init_qpos: tuple[float, ...]
    key_qpos: tuple[tuple[float, ...], ...]
    key_qvel: tuple[tuple[float, ...], ...]
    jnt_range: tuple[tuple[float, float], ...]
    actuator_names: tuple[str, ...]
    actuator_targets: dict[int, tuple[str, ...]]
    na: int


def _model_key(task: CpuTaskSpec) -> tuple[Any, ...]:
    """Cache key: models only differ by path/recipe/edit_fn (not by condition)."""
    kw = task.kwargs
    return (kw.get("model_path"), kw.get("model_recipe"), kw.get("edit_fn"))


@functools.cache
def _compiled_info(key: tuple[Any, ...]) -> CompiledModelInfo:
    model_path, model_recipe, edit_fn = key
    kwargs = {"model_path": model_path, "model_recipe": model_recipe}
    if edit_fn is not None:
        kwargs["edit_fn"] = edit_fn
    model = build_cpu_spec(CpuTaskSpec("", kwargs, 0)).compile()
    return CompiledModelInfo(
        opt_timestep=float(model.opt.timestep),
        mujoco_cfg=mujoco_cfg_from_model(model),
        joint_names=tuple(model.joint(i).name for i in range(model.njnt)),
        jnt_type=tuple(int(t) for t in model.jnt_type),
        jnt_qposadr=tuple(int(a) for a in model.jnt_qposadr),
        init_qpos=tuple(float(q) for q in _cpu_init_qpos(model)),
        key_qpos=tuple(tuple(float(q) for q in k) for k in model.key_qpos),
        key_qvel=tuple(tuple(float(v) for v in k) for k in model.key_qvel),
        jnt_range=tuple((float(lo), float(hi)) for lo, hi in model.jnt_range),
        actuator_names=tuple(model.actuator(i).name for i in range(model.nu)),
        actuator_targets=_actuator_targets(model),
        na=int(model.na),
    )


def _actuator_targets(model: mujoco.MjModel) -> dict[int, tuple[str, ...]]:
    """Names of the joints / tendons driven by actuators, keyed by ``mjtTrn``."""
    obj = {
        int(mujoco.mjtTrn.mjTRN_JOINT): mujoco.mjtObj.mjOBJ_JOINT,
        int(mujoco.mjtTrn.mjTRN_TENDON): mujoco.mjtObj.mjOBJ_TENDON,
    }
    targets: dict[int, list[str]] = {}
    for trn_type, trn_id in zip(model.actuator_trntype, model.actuator_trnid[:, 0]):
        if int(trn_type) in obj:
            name = mujoco.mj_id2name(model, obj[int(trn_type)], int(trn_id))
            targets.setdefault(int(trn_type), []).append(name)
    return {k: tuple(v) for k, v in targets.items()}


def _cpu_init_qpos(model: mujoco.MjModel) -> np.ndarray:
    """CPU ``_init_qpos`` for ``normalize_act=True``: ``qpos0``, except that
    joint-actuated hinge/slide joints start at the middle of their range."""
    init_qpos = model.qpos0.copy()
    actuated = model.actuator_trnid[
        model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
    ]
    linear = np.where(
        np.logical_or(
            model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
            model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
        )
    )[0]
    ids = np.intersect1d(actuated, linear)
    init_qpos[model.jnt_qposadr[ids]] = np.mean(model.jnt_range[ids], axis=1)
    return init_qpos


def compiled_info(task: CpuTaskSpec) -> CompiledModelInfo:
    """Compiled-model constants of *task* (cached per model)."""
    return _compiled_info(_model_key(task))


def mujoco_cfg_from_model(model: mujoco.MjModel) -> MujocoCfg:
    """Copy the XML ``<option>`` of *model* into an mjlab ``MujocoCfg``.

    mjlab overwrites ``model.opt`` with its ``MujocoCfg`` (default integrator
    ``implicitfast``), so the CPU physics settings must be carried over
    explicitly.

    Args:
        model: The compiled CPU model.

    Returns:
        A ``MujocoCfg`` reproducing the model's solver/integrator options.

    Raises:
        ValueError: If the model uses an integrator MuJoCo Warp does not support.
    """
    opt = model.opt
    if int(opt.integrator) not in _INTEGRATORS:
        raise ValueError(
            f"Integrator {mujoco.mjtIntegrator(opt.integrator).name} is not "
            "supported by MuJoCo Warp; the mjlab port cannot match the CPU physics."
        )
    disable = tuple(
        name.removeprefix("mjDSBL_").lower()
        for name, bit in mujoco.mjtDisableBit.__members__.items()
        if name != "mjNDISABLE" and opt.disableflags & int(bit)
    )
    enable = tuple(
        name.removeprefix("mjENBL_").lower()
        for name, bit in mujoco.mjtEnableBit.__members__.items()
        if name != "mjNENABLE" and opt.enableflags & int(bit)
    )
    return MujocoCfg(
        timestep=float(opt.timestep),
        integrator=_INTEGRATORS[int(opt.integrator)],  # type: ignore[arg-type]
        impratio=float(opt.impratio),
        cone=_CONES[int(opt.cone)],  # type: ignore[arg-type]
        jacobian=_JACOBIANS[int(opt.jacobian)],  # type: ignore[arg-type]
        solver=_SOLVERS[int(opt.solver)],  # type: ignore[arg-type]
        iterations=int(opt.iterations),
        tolerance=float(opt.tolerance),
        ls_iterations=int(opt.ls_iterations),
        ls_tolerance=float(opt.ls_tolerance),
        ccd_iterations=int(opt.ccd_iterations),
        gravity=tuple(float(g) for g in opt.gravity),  # type: ignore[arg-type]
        disableflags=disable,
        enableflags=enable,
    )


def init_state_from_model(
    info: CompiledModelInfo, qpos: tuple[float, ...] | None = None
) -> EntityCfg.InitialStateCfg:
    """Initial state at a CPU ``qpos`` (default: the CPU ``_init_qpos``), zero velocity.

    A free root joint (``qpos`` address 0) maps to the entity root pose; every hinge/slide joint is
    pinned to its value. Tasks whose CPU reset writes a non-zero ``qvel`` (e.g.
    keyframe resets) restore the exact state with a reset event.

    Args:
        info: Compiled-model constants.
        qpos: Full CPU-layout ``qpos``.

    Returns:
        mjlab initial-state config.

    Raises:
        ValueError: For ball joints (no scalar ``joint_pos`` entry).
    """
    qpos = info.init_qpos if qpos is None else qpos
    root: dict[str, tuple[float, ...]] = {}
    joint_pos: dict[str, float] = {}
    for name, jtype, adr in zip(
        info.joint_names, info.jnt_type, info.jnt_qposadr, strict=True
    ):
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            if adr != 0:  # not the entity root; see demote_extra_freejoints
                continue
            root = {
                "pos": tuple(qpos[adr : adr + 3]),
                "rot": tuple(qpos[adr + 3 : adr + 7]),
            }
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            raise ValueError(f"Ball joint {name!r} is not supported.")
        else:
            joint_pos[f"^{name}$"] = qpos[adr]
    return EntityCfg.InitialStateCfg(**root, joint_pos=joint_pos, joint_vel={".*": 0.0})


def _entity_spec(
    task: CpuTaskSpec, spec_edits: tuple[Callable[[mujoco.MjSpec], None], ...]
) -> mujoco.MjSpec:
    """CPU spec + task edits, keyframes removed (unnamed XML keys clash on attach;
    their values live in :class:`CompiledModelInfo` and reset events)."""
    spec = build_cpu_spec(task)
    for edit in spec_edits:
        edit(spec)
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def demote_extra_freejoints(spec: mujoco.MjSpec) -> None:
    """Replace every freejoint mjlab cannot use as entity root by a 6-DoF chain.

    An mjlab entity has at most one freejoint, and it must be the spec's first
    joint. Any other freejoint (e.g. the passive, soft-welded exosuit parts of
    ``myotorso_exosuit.xml``) becomes three world-axis slides plus three hinges at
    the body origin, starting from the body's own pose. Dynamics match; the ``qpos``
    of such a body holds 6 Euler-chain values instead of a 7-value pose, so the
    observation is one value shorter per demoted joint than on CPU.

    Args:
        spec: Entity spec, edited in place.
    """
    joints = list(spec.joints)
    for joint in joints:
        if joint.type != mujoco.mjtJoint.mjJNT_FREE or joint is joints[0]:
            continue
        body = joint.parent
        spec.delete(joint)
        for kind, axes in (
            (mujoco.mjtJoint.mjJNT_SLIDE, np.eye(3)),
            (mujoco.mjtJoint.mjJNT_HINGE, np.eye(3)),
        ):
            for axis in axes:
                name = f"{body.name}_{'xyz'[int(np.argmax(axis))]}_{kind.name[-5:].lower()}"
                body.add_joint(name=name, type=kind, axis=axis)


_XML_TRANSMISSIONS = {
    int(mujoco.mjtTrn.mjTRN_JOINT): TransmissionType.JOINT,
    int(mujoco.mjtTrn.mjTRN_TENDON): TransmissionType.TENDON,
}


def robot_entity_cfg(
    task: CpuTaskSpec,
    init_qpos: tuple[float, ...] | None = None,
    spec_edits: tuple[Callable[[mujoco.MjSpec], None], ...] = (),
) -> EntityCfg:
    """Entity built from the CPU model, its XML actuators wrapped as-is.

    Args:
        task: CPU registration of the task.
        init_qpos: CPU-layout default ``qpos`` (default: CPU ``_init_qpos``).
        spec_edits: Edits the CPU env applies to its compiled model at init
            (e.g. hiding the terrain), replayed on the spec.

    Returns:
        Entity config starting from the CPU reset pose.
    """
    info = compiled_info(task)
    actuators = tuple(
        XmlActuatorCfg(
            target_names_expr=tuple(f"^{n}$" for n in info.actuator_targets[trn_type]),
            transmission_type=trn,
        )
        for trn_type, trn in _XML_TRANSMISSIONS.items()
        if trn_type in info.actuator_targets
    )
    return EntityCfg(
        spec_fn=functools.partial(_entity_spec, task, spec_edits),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
        init_state=init_state_from_model(info, init_qpos),
    )


def action_cfg(
    task: CpuTaskSpec,
    entity_name: str,
    action_range: tuple[float, float] = (-1.0, 1.0),
    muscle_sigmoid: bool = True,
) -> MyoActionCfg:
    """CPU action pipeline of *task* (normalization + muscle condition).

    Reafferentation reroutes EIP's command to EPL and silences EIP (``_r``
    suffix on recipe-built models), exactly as the CPU envs do.

    Args:
        task: CPU registration of the task.
        entity_name: Scene entity receiving the actions.
        action_range: Normalized action space of the CPU env class.
        muscle_sigmoid: Whether the CPU env class maps muscle actions through
            the sigmoid (``LegWalkEnvV0`` does not).

    Returns:
        The action term config.
    """
    reroute = None
    if task.muscle_condition == "reafferentation":
        sfx = "_r" if task.uses_recipe else ""
        reroute = (f"EIP{sfx}", f"EPL{sfx}")
    return MyoActionCfg(
        entity_name=entity_name,
        normalize_act=bool(task.kwargs.get("normalize_act", True)),
        action_range=action_range,
        muscle_sigmoid=muscle_sigmoid,
        muscle_fatigue=task.muscle_condition == "fatigue",
        reroute=reroute,
    )


def episode_length_s(max_episode_steps: int, step_dt: float) -> float:
    """Episode length whose ``ceil(len / step_dt)`` equals *max_episode_steps*."""
    return (max_episode_steps - 0.5) * step_dt


def first_step_after(time_s: float, timestep: float, frame_skip: int) -> int:
    """First control step at which the CPU ``data.time`` exceeds *time_s*.

    CPU envs gate some terms on ``data.time > t`` where ``data.time`` is a
    float64 sum of ``timestep`` increments; replaying that sum reproduces the
    exact boundary step instead of relying on float equality.

    Args:
        time_s: Threshold time in seconds.
        timestep: Physics timestep.
        frame_skip: Substeps per control step.

    Returns:
        Smallest control-step count ``k`` with accumulated time ``> time_s``.
    """
    t = 0.0
    step = 0
    while t <= time_s:
        for _ in range(frame_skip):
            t += timestep
        step += 1
    return step
