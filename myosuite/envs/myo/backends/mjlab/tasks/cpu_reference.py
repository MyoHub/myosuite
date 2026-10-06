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

import contextlib
import contextvars
import functools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.registration import load_env_creator
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.sim import MujocoCfg

from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg
from myosuite.terms.base_action import MotorNoiseCfg

_CONDITION_WRAPPERS = {
    "sarcopenia": "SarcopeniaWrapper",
    "fatigue": "FatigueWrapper",
    "reafferentation": "ReafferentationWrapper",
}
_MUSCLE_WRAPPERS = (
    *_CONDITION_WRAPPERS.values(),
    "MotorNoiseWrapper",
    "ExcitationStageWrapper",
)

# Wrapper specs of the current ``make_env(EnvConfig(features=...))`` call, added to the
# registered ones by :func:`cpu_task_spec` (every twin factory reads the registration there).
_FEATURES: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar(
    "myosuite_features", default=()
)
_KWARGS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "myosuite_task_kwargs", default={}
)


@contextlib.contextmanager
def feature_overrides(
    features: Iterable[Any] = (), task_kwargs: dict[str, Any] | None = None
) -> Iterator[None]:
    """Change what every :func:`cpu_task_spec` read inside the block returns.

    Args:
        features: :class:`~gymnasium.envs.registration.WrapperSpec` of the muscle-command
            wrappers (``EnvConfig.features``), added to the registered ones.
        task_kwargs: CPU env constructor kwargs (such as ``frame_skip``) that replace
            the registered ones.
    """
    tokens = (_FEATURES.set(tuple(features)), _KWARGS.set(dict(task_kwargs or {})))
    try:
        yield
    finally:
        _FEATURES.reset(tokens[0])
        _KWARGS.reset(tokens[1])


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
        wrappers: The wrapper specs of the registration; the muscle-command
            wrappers (:mod:`myosuite.envs.wrappers`) configure the twin's action
            pipeline.
    """

    env_id: str
    kwargs: dict[str, Any]
    max_episode_steps: int
    wrappers: tuple[Any, ...] = ()

    def wrapper_specs(self, name: str) -> list[dict[str, Any]]:
        """Constructor kwargs of every registered wrapper called *name*."""
        return [dict(s.kwargs or {}) for s in self.wrappers if s.name == name]

    def wrapper_kwargs(self, name: str) -> dict[str, Any] | None:
        """Constructor kwargs of the registered wrapper *name*, or ``None`` if absent."""
        for spec in self.wrappers:
            if spec.name == name:
                return dict(spec.kwargs or {})
        return None

    @property
    def muscle_conditions(self) -> tuple[str, ...]:
        """Registered conditions (``"sarcopenia"``, ``"fatigue"``, ``"reafferentation"``).

        The condition wrappers compose, so every one of them configures the twin.
        """
        return tuple(
            condition
            for condition, name in _CONDITION_WRAPPERS.items()
            if self.wrapper_kwargs(name) is not None
        )

    @property
    def motor_noise(self) -> MotorNoiseCfg:
        """Noise on muscle excitations (``MotorNoiseWrapper``; off by default)."""
        kwargs = self.wrapper_kwargs("MotorNoiseWrapper") or {}
        return MotorNoiseCfg.from_value(kwargs.get("motor_noise"))

    @property
    def fatigue_reset(self) -> tuple[Any, bool]:
        """``(fatigue_reset_vec, fatigue_reset_random)`` of the ``FatigueWrapper``."""
        kwargs = self.wrapper_kwargs("FatigueWrapper") or {}
        return kwargs.get("fatigue_reset_vec"), bool(
            kwargs.get("fatigue_reset_random", False)
        )

    @property
    def frame_skip(self) -> int:
        """Physics substeps per control step (CPU default 10)."""
        return int(self.kwargs.get("frame_skip", 10))

    @property
    def uses_recipe(self) -> bool:
        """Whether the model comes from a ``ModelBuilder`` recipe (``_r`` names)."""
        return self.kwargs.get("model_recipe") is not None


def _with_features(registered: tuple[Any, ...], env_id: str) -> tuple[Any, ...]:
    """The registered wrapper specs plus the ``EnvConfig.features`` of this call."""
    added = _FEATURES.get()
    for spec in added:
        if spec.name != "ExcitationStageWrapper" and any(
            r.name == spec.name for r in registered
        ):
            raise ValueError(
                f"{env_id} already has a {spec.name}; apply each wrapper once "
                "(use the base id, as on the CPU)."
            )
    return registered + added


def cpu_task_spec(env_id: str) -> CpuTaskSpec:
    """Read the CPU registration of *env_id*.

    Args:
        env_id: A registered MyoSuite CPU env id.

    Returns:
        The frozen registration data.

    Raises:
        ValueError: If the registration has muscle-command wrappers on a CPU env
            class that does not run them (the twin would not match).
    """
    import myosuite  # noqa: F401, PLC0415  (registers the CPU envs)

    spec = gym.spec(env_id)
    task = CpuTaskSpec(
        env_id=env_id,
        kwargs={**spec.kwargs, **_KWARGS.get()},
        max_episode_steps=int(spec.max_episode_steps),
        wrappers=_with_features(tuple(spec.additional_wrappers or ()), env_id),
    )
    if any(task.wrapper_kwargs(name) is not None for name in _MUSCLE_WRAPPERS):
        entry = spec.entry_point
        env_cls = load_env_creator(entry) if isinstance(entry, str) else entry
        if not getattr(env_cls, "supports_ctrl_stages", False):
            raise ValueError(
                f"{env_id} registers muscle-command wrappers on {env_cls.__name__}, "
                "whose action pipeline does not run them."
            )
    return task


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
    sarcopenia = task.wrapper_kwargs("SarcopeniaWrapper")
    if sarcopenia is not None:
        apply_sarcopenia_to_spec(spec, force_scale=sarcopenia.get("force_scale", 0.5))
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
    qpos0: tuple[float, ...]
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
        qpos0=tuple(float(q) for q in model.qpos0),
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


def mujoco_cfg_from_model(
    model: mujoco.MjModel, timestep: float | None = None
) -> MujocoCfg:
    """Copy the XML ``<option>`` of *model* into an mjlab ``MujocoCfg``.

    mjlab overwrites ``model.opt`` with its ``MujocoCfg`` (default integrator
    ``implicitfast``), so the CPU physics settings must be carried over
    explicitly.

    Args:
        model: The compiled CPU model.
        timestep: Physics step the caller derived its decimation from. If given,
            it must equal the model's, so the control step cannot drift from CPU.

    Returns:
        A ``MujocoCfg`` reproducing the model's solver/integrator options.

    Raises:
        ValueError: If the model uses an integrator MuJoCo Warp does not support,
            or if *timestep* differs from the model's.
    """
    opt = model.opt
    if timestep is not None and timestep != float(opt.timestep):
        raise ValueError(
            f"Config timestep {timestep} differs from the CPU model's "
            f"{float(opt.timestep)}; the control step would not match CPU."
        )
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


@functools.cache
def _musclemimic_cpu_model(variant: str) -> mujoco.MjModel:
    from myosuite.core.model_recipes import _musclemimic_build  # noqa: PLC0415

    return _musclemimic_build(f"musclemimic_{variant}")[0]


def musclemimic_mujoco_cfg(variant: str, timestep: float | None = None) -> MujocoCfg:
    """``MujocoCfg`` of the CPU MuscleMimic model of *variant*.

    The CPU envs (``myoMimicBimanual-v0`` / ``myoMimicFullbody-v0``) compile
    with ``compile_mimic_*_mjmodel``, which edits ``model.opt`` after compiling
    the spec (``sim_dt``; bimanual also solver iterations and ``eulerdamp``),
    so the spec's own ``<option>`` is not the CPU physics.

    Args:
        variant: ``"bimanual"`` or ``"fullbody"``.
        timestep: See :func:`mujoco_cfg_from_model`.

    Returns:
        A ``MujocoCfg`` reproducing the CPU model's options.
    """
    return mujoco_cfg_from_model(_musclemimic_cpu_model(variant), timestep=timestep)


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


@dataclass(frozen=True)
class FreeJointChain:
    """A freejoint replaced by a 6-DoF chain (:func:`demote_extra_freejoints`).

    Attributes:
        chain_start: Index of the chain's first slide in the entity's ``joint_pos``.
        pos0: Rest position of the body (its ``qpos0``).
        quat0: Rest orientation ``wxyz`` of the body (its ``qpos0``).
    """

    chain_start: int
    pos0: tuple[float, ...]
    quat0: tuple[float, ...]


def free_joint_chains(info: CompiledModelInfo) -> tuple[FreeJointChain, ...]:
    """Chains :func:`demote_extra_freejoints` creates for the model of *info*."""
    chains: list[FreeJointChain] = []
    for jtype, adr in zip(info.jnt_type, info.jnt_qposadr, strict=True):
        if jtype == mujoco.mjtJoint.mjJNT_FREE and adr != 0:
            chains.append(
                FreeJointChain(
                    adr - len(chains),  # each earlier chain has 6 values, not 7
                    tuple(info.qpos0[adr : adr + 3]),
                    tuple(info.qpos0[adr + 3 : adr + 7]),
                )
            )
    return tuple(chains)


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
    ``myotorso_exosuit.xml``) becomes three slides (along the body's rest-frame axes)
    plus three hinges (x, y, z) at the body origin, starting from the body's own
    pose. Dynamics match; the ``qpos`` of such a body holds 6 chain values instead of
    a 7-value pose. The observation terms ``qpos_chains`` / ``qvel_chains`` convert
    them back to the CPU layout (see :class:`FreeJointChain`).

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
    """CPU action pipeline of *task* (normalization, motor noise, muscle condition).

    Reafferentation reroutes EIP's command to EPL and silences EIP (``_r``
    suffix on recipe-built models), exactly as the CPU envs do. Fatigue
    carries the CPU reset options ``fatigue_reset_vec`` / ``fatigue_reset_random``.

    Args:
        task: CPU registration of the task.
        entity_name: Scene entity receiving the actions.
        action_range: Normalized action space of the CPU env class.
        muscle_sigmoid: Whether the CPU env class maps muscle actions through
            the sigmoid (``LegWalkEnvV0`` does not).

    Returns:
        The action term config.
    """
    conditions = task.muscle_conditions
    reroute = None
    if "reafferentation" in conditions:
        sfx = "_r" if task.uses_recipe else ""
        reroute = (f"EIP{sfx}", f"EPL{sfx}")
    fatigue = "fatigue" in conditions
    # The CPU fatigue reset options (only read by the CPU env under fatigue).
    reset_vec, reset_random = task.fatigue_reset if fatigue else (None, False)
    return MyoActionCfg(
        entity_name=entity_name,
        normalize_act=bool(task.kwargs.get("normalize_act", True)),
        action_range=action_range,
        muscle_sigmoid=muscle_sigmoid,
        muscle_fatigue=fatigue,
        fatigue_reset_vec=(
            None if reset_vec is None else tuple(float(v) for v in reset_vec)
        ),
        fatigue_reset_random=reset_random,
        reroute=reroute,
        motor_noise=task.motor_noise,
        excitation_stages=tuple(
            spec["make_stage"] for spec in task.wrapper_specs("ExcitationStageWrapper")
        ),
    )


def episode_length_s(max_episode_steps: int, step_dt: float) -> float:
    """Episode length whose ``ceil(len / step_dt)`` equals *max_episode_steps*."""
    return (max_episode_steps - 0.5) * step_dt
