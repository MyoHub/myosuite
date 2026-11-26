# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental Newton ``SolverMuJoCo`` index mapping and ``EnvAccessor`` adapter.

.. warning::
   **Experimental validation probe only.** CPU ``MyoGymnasiumEnv`` and GPU
   ``mjlab`` are the only two supported MyoSuite backends. Like the experimental
   MJX route, this module is intended for solver compatibility probes rather than
   new task development.
"""

from __future__ import annotations

import importlib
import inspect
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.core.model_recipes import materialize_recipe_xml
from myosuite.core.protocols import EnvAccessor, PhysicsPath
from myosuite.utils.asset_path_resolver import resolve_model_xml_path


class _NewtonSimContext(Protocol):
    reference_model: mujoco.MjModel
    solver: Any
    mapping: IndexMapping
    num_envs: int
    ctrl_dt: float


def _require_newton() -> tuple[Any, Any]:
    try:
        import newton
        import warp as wp
    except ImportError as exc:
        raise ImportError(
            "newton>=1.8.0 and warp-lang are required for myosuite.integrations.newton."
        ) from exc
    return newton, wp


def _integrator_name(model: mujoco.MjModel) -> str:
    return {
        int(mujoco.mjtIntegrator.mjINT_EULER): "euler",
        int(mujoco.mjtIntegrator.mjINT_RK4): "rk4",
        int(mujoco.mjtIntegrator.mjINT_IMPLICIT): "implicit",
        int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
    }[int(model.opt.integrator)]


def _entry_point_defaults(entry_point: Any) -> dict[str, Any]:
    target = entry_point
    if isinstance(entry_point, str) and ":" in entry_point:
        mod_name, attr_name = entry_point.split(":", 1)
        target = getattr(importlib.import_module(mod_name), attr_name)
    if not callable(target):
        return {}
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return {}
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def resolve_env_xml(
    env_id_or_xml: str | Path, workdir: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Resolve an MJCF path and metadata from an XML path or registered env ID.

    Args:
        env_id_or_xml: Existing MJCF file path or registered MyoSuite env ID.
        workdir: Optional directory used when materializing recipe-based models.

    Returns:
        Tuple ``(xml_path, meta)`` where ``meta`` holds ``frame_skip``,
        ``normalize_act``, and ``max_episode_steps`` when resolved from an env ID.
    """
    candidate = Path(env_id_or_xml)
    if candidate.suffix.lower() == ".xml" or candidate.exists():
        return Path(resolve_model_xml_path(str(candidate))), {}

    import myosuite  # noqa: F401

    spec = gym.spec(str(env_id_or_xml))
    kwargs = dict(spec.kwargs)
    defaults = _entry_point_defaults(spec.entry_point)
    entry_str = str(spec.entry_point)
    sarc_scale = None
    for w in spec.additional_wrappers or ():
        if w.name == "SarcopeniaWrapper":
            sarc_scale = float((w.kwargs or {}).get("force_scale", 0.5))
            break
    meta = {
        "frame_skip": int(kwargs.get("frame_skip", defaults.get("frame_skip", 10))),
        "normalize_act": bool(
            kwargs.get("normalize_act", defaults.get("normalize_act", True))
        ),
        "max_episode_steps": spec.max_episode_steps,
        "hide_terrain": (
            ("LegWalkEnvV0" in entry_str or "LegReachEnvV0" in entry_str)
            and not kwargs.get("terrain")
        ),
        "muscle_sigmoid": not (
            "LegWalkEnv" in entry_str or "walk" in entry_str.lower()
        ),
        "sarcopenia_force_scale": sarc_scale,
    }
    recipe = kwargs.get("model_recipe")
    if recipe is not None:
        base_dir = workdir or (Path(tempfile.gettempdir()) / "myosuite_newton_xml")
        base_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w.-]", "_", str(env_id_or_xml))
        return (
            materialize_recipe_xml(
                recipe, base_dir / f"{slug}.xml", edit_fn=kwargs.get("edit_fn")
            ),
            meta,
        )
    if kwargs.get("edit_fn") is not None:
        raise ValueError(
            f"{env_id_or_xml!r} applies edit_fn to model_path and cannot be exported statically."
        )
    return Path(resolve_model_xml_path(kwargs["model_path"])), meta


def _merged_dof_sources(reference: mujoco.MjModel, newton_model: Any) -> dict[int, int]:
    """Recover source joint IDs when Newton merges multi-joint bodies."""
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    merged: dict[str, list[int]] = {}
    for body in range(reference.nbody):
        start = int(reference.body_jntadr[body])
        rows = range(start, start + int(reference.body_jntnum[body]))
        if len(rows) > 1:
            name = "_".join(reference.joint(row).name for row in rows)
            merged[name] = sorted(
                rows, key=lambda r: int(reference.jnt_type[r]) != slide
            )
    starts = newton_model.joint_qd_start.numpy()
    joint_per_world = len(newton_model.joint_label) // max(
        1, int(newton_model.world_count)
    )
    sources: dict[int, int] = {}
    for joint in range(joint_per_world):
        label = newton_model.joint_label[joint]
        rows = merged.get(label.rsplit("/", 1)[-1], [])
        sources.update(zip(range(int(starts[joint]), int(starts[joint + 1])), rows))
    return sources


@dataclass(frozen=True)
class IndexMapping:
    """Bidirectional index permutations between authored MuJoCo and Newton SolverMuJoCo."""

    conv_to_src_qpos: np.ndarray
    src_to_conv_qpos: np.ndarray
    conv_to_src_qvel: np.ndarray
    src_to_conv_qvel: np.ndarray
    conv_to_src_ctrl: np.ndarray
    src_to_conv_ctrl: np.ndarray
    src_to_conv_act: np.ndarray
    src_to_conv_site: np.ndarray
    src_to_conv_body: np.ndarray = None  # type: ignore[assignment]

    @classmethod
    def build(
        cls, reference: mujoco.MjModel, newton_model: Any, solver: Any
    ) -> IndexMapping:
        """Build exact index permutations between ``reference`` and ``solver.mj_model``."""
        converted = solver.mj_model
        labels = newton_model.mujoco.joint_dof_label
        dofs = solver.mjc_jnt_to_newton_dof.numpy()[0]
        joints = solver.mjc_jnt_to_newton_jnt.numpy()[0]
        source_jnt = {reference.joint(r).name: r for r in range(reference.njnt)}
        merged = _merged_dof_sources(reference, newton_model)

        conv_to_src_qpos = np.full(converted.nq, -1, dtype=np.intp)
        conv_to_src_qvel = np.full(converted.nv, -1, dtype=np.intp)
        for row in range(converted.njnt):
            dof = int(dofs[row])
            if dof >= 0 and labels[dof]:
                orig = source_jnt[labels[dof].rsplit("/", 1)[-1]]
            elif dof in merged:
                orig = merged[dof]
            else:
                orig = source_jnt[
                    newton_model.joint_label[int(joints[row])].rsplit("/", 1)[-1]
                ]
            jtype = int(converted.jnt_type[row])
            q_w = {
                int(mujoco.mjtJoint.mjJNT_FREE): 7,
                int(mujoco.mjtJoint.mjJNT_BALL): 4,
            }.get(jtype, 1)
            v_w = {
                int(mujoco.mjtJoint.mjJNT_FREE): 6,
                int(mujoco.mjtJoint.mjJNT_BALL): 3,
            }.get(jtype, 1)
            cq, sq = int(converted.jnt_qposadr[row]), int(reference.jnt_qposadr[orig])
            cv, sv = int(converted.jnt_dofadr[row]), int(reference.jnt_dofadr[orig])
            conv_to_src_qpos[cq : cq + q_w] = np.arange(sq, sq + q_w)
            conv_to_src_qvel[cv : cv + v_w] = np.arange(sv, sv + v_w)

        src_to_conv_qpos = np.argsort(conv_to_src_qpos)
        src_to_conv_qvel = np.argsort(conv_to_src_qvel)

        if converted.nu == 0:
            conv_to_src_ctrl = np.empty(0, dtype=np.intp)
            src_to_conv_ctrl = np.empty(0, dtype=np.intp)
        else:
            act_labels = newton_model.mujoco.actuator_label
            act_map = getattr(
                solver,
                "mjc_actuator_to_newton_actuator",
                solver.mjc_actuator_to_newton_actuator_idx,
            )
            act_indices = act_map.numpy()
            if act_indices.ndim > 1:
                act_indices = act_indices[0]
            conv_to_src_ctrl = np.array(
                [
                    reference.actuator(act_labels[int(idx)].rsplit("/", 1)[-1]).id
                    for idx in act_indices[: converted.nu]
                ],
                dtype=np.intp,
            )
            src_to_conv_ctrl = np.argsort(conv_to_src_ctrl)

        src_to_conv_act = np.full(reference.na, -1, dtype=np.intp)
        for conv_u, src_u in enumerate(conv_to_src_ctrl):
            src_adr = int(reference.actuator_actadr[src_u])
            conv_adr = int(converted.actuator_actadr[conv_u])
            if src_adr >= 0 and conv_adr >= 0:
                num = int(reference.actuator_actnum[src_u])
                src_to_conv_act[src_adr : src_adr + num] = np.arange(
                    conv_adr, conv_adr + num
                )

        if converted.nsite == 0:
            conv_to_src_site = np.empty(0, dtype=np.intp)
            src_to_conv_site = np.empty(0, dtype=np.intp)
        else:
            site_map = getattr(
                solver, "mjc_site_to_newton_shape", solver._mjc_site_to_newton_shape
            )
            site_shapes = site_map.numpy()[0]
            raw_shape_labels = getattr(newton_model.mujoco, "shape_label", None)
            conv_to_src_site = np.array(
                [
                    reference.site(
                        raw_shape_labels[int(s)]
                        if (raw_shape_labels is not None and raw_shape_labels[int(s)])
                        else newton_model.shape_label[int(s)].rsplit("/", 1)[-1]
                    ).id
                    for s in site_shapes[: converted.nsite]
                ],
                dtype=np.intp,
            )
            src_to_conv_site = np.argsort(conv_to_src_site)

        src_to_conv_body = np.zeros(reference.nbody, dtype=np.intp)
        body_map = solver.mjc_body_to_newton.numpy()[0]
        raw_body_labels = getattr(newton_model.mujoco, "body_label", None)
        for conv_b in range(1, converted.nbody):
            nb = int(body_map[conv_b])
            if nb < 0:
                continue
            b_name = (
                raw_body_labels[nb]
                if (raw_body_labels is not None and raw_body_labels[nb])
                else newton_model.body_label[nb].rsplit("/", 1)[-1]
            )
            src_to_conv_body[reference.body(b_name).id] = conv_b

        return cls(
            conv_to_src_qpos=conv_to_src_qpos,
            src_to_conv_qpos=src_to_conv_qpos,
            conv_to_src_qvel=conv_to_src_qvel,
            src_to_conv_qvel=src_to_conv_qvel,
            conv_to_src_ctrl=conv_to_src_ctrl,
            src_to_conv_ctrl=src_to_conv_ctrl,
            src_to_conv_act=src_to_conv_act,
            src_to_conv_site=src_to_conv_site,
            src_to_conv_body=src_to_conv_body,
        )


class NewtonAccessor(EnvAccessor):
    """Batched EnvAccessor presenting Newton SolverMuJoCo state in authored MuJoCo order."""

    def __init__(self, sim: _NewtonSimContext) -> None:
        self._sim = sim

    @property
    def physics_path(self) -> PhysicsPath:
        return PhysicsPath.CPU

    def joint_pos(self) -> np.ndarray:
        qpos = self._sim.solver.mjw_data.qpos.numpy()
        return qpos[:, self._sim.mapping.src_to_conv_qpos].copy()

    def joint_vel(self) -> np.ndarray:
        qvel = self._sim.solver.mjw_data.qvel.numpy()
        return qvel[:, self._sim.mapping.src_to_conv_qvel].copy()

    def muscle_act(self) -> np.ndarray:
        if self._sim.reference_model.na == 0:
            return np.zeros((self._sim.num_envs, 0), dtype=np.float32)
        act = self._sim.solver.mjw_data.act.numpy()
        return act[:, self._sim.mapping.src_to_conv_act].copy()

    def site_xpos(self, site_ids: Any) -> np.ndarray:
        xpos = self._sim.solver.mjw_data.site_xpos.numpy()
        conv_ids = self._sim.mapping.src_to_conv_site[site_ids]
        return xpos[:, conv_ids].copy()

    def site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(
            self._sim.reference_model, mujoco.mjtObj.mjOBJ_SITE, name
        )
        if sid == -1:
            raise ValueError(f"Site {name!r} not found in model")
        return int(sid)

    def time(self) -> np.ndarray:
        return self._sim.solver.mjw_data.time.numpy().copy()

    def ctrl_range(self) -> np.ndarray:
        return self._sim.reference_model.actuator_ctrlrange.astype(np.float32).copy()

    def dt(self) -> float:
        return self._sim.ctrl_dt

    def array_module(self) -> Any:
        return np

    def muscle_length(self) -> np.ndarray:
        length = self._sim.solver.mjw_data.actuator_length.numpy()
        return length[:, self._sim.mapping.src_to_conv_ctrl].copy()

    def muscle_velocity(self) -> np.ndarray:
        vel = self._sim.solver.mjw_data.actuator_velocity.numpy()
        return vel[:, self._sim.mapping.src_to_conv_ctrl].copy()

    def muscle_force(self) -> np.ndarray:
        frc = self._sim.solver.mjw_data.actuator_force.numpy()
        return frc[:, self._sim.mapping.src_to_conv_ctrl].copy()

    def body_xpos(self) -> np.ndarray:
        xpos = self._sim.solver.mjw_data.xpos.numpy()
        return xpos[:, self._sim.mapping.src_to_conv_body].copy()

    def body_xquat(self) -> np.ndarray:
        xquat = self._sim.solver.mjw_data.xquat.numpy()
        return xquat[:, self._sim.mapping.src_to_conv_body].copy()

    def body_xipos(self) -> np.ndarray:
        xipos = self._sim.solver.mjw_data.xipos.numpy()
        return xipos[:, self._sim.mapping.src_to_conv_body].copy()

    def body_cvel(self) -> np.ndarray:
        cvel = self._sim.solver.mjw_data.cvel.numpy()
        return cvel[:, self._sim.mapping.src_to_conv_body].copy()

    def subtree_com(self) -> np.ndarray:
        com = self._sim.solver.mjw_data.subtree_com.numpy()
        return com[:, self._sim.mapping.src_to_conv_body].copy()

    def contact_counts(self) -> np.ndarray:
        nacon = int(self._sim.solver.mjw_data.nacon.numpy()[0])
        if nacon <= 0:
            return np.zeros(self._sim.num_envs, dtype=np.int32)
        worldid = self._sim.solver.mjw_data.contact.worldid.numpy()[:nacon]
        valid = worldid[(worldid >= 0) & (worldid < self._sim.num_envs)]
        return np.bincount(valid, minlength=self._sim.num_envs).astype(np.int32)
