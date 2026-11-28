# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import mujoco


def compile_with_options(spec: mujoco.MjSpec, model: mujoco.MjModel) -> mujoco.MjModel:
    """Recompile *spec*, keeping the runtime options set on *model*.

    Envs often set ``model.opt`` (timestep, solver iterations, flags) after
    compiling, so a plain ``spec.compile()`` would silently revert them to the
    XML values. Every ``mjOption`` field is copied into ``spec.option`` first.

    Args:
        spec: Spec to compile; its ``option`` is overwritten from *model*.
        model: Previously compiled model whose ``opt`` is kept.

    Returns:
        The newly compiled model.
    """
    for name in dir(model.opt):
        if name.startswith("_"):
            continue
        value = getattr(model.opt, name)
        if not callable(value):
            setattr(spec.option, name, value)
    return spec.compile()


def recursive_immobilize(
    spec: mujoco.MjSpec,
    temp_model: mujoco.MjModel | None,
    parent: Any,
    remove_eqs: bool = False,
    remove_actuators: bool = False,
    remove_sites: bool = True,
) -> list[int]:
    removed_joint_ids: list[int] = []
    if remove_sites:
        for s in list(parent.sites):
            spec.delete(s)
    for j in list(parent.joints):
        jname = j.name
        if temp_model is not None:
            removed_joint_ids.extend(temp_model.joint(jname).qposadr)
        if remove_eqs:
            for e in list(spec.equalities):
                if e.type == mujoco.mjtEq.mjEQ_JOINT and (
                    e.name1 == jname or e.name2 == jname
                ):
                    spec.delete(e)
            removed_tendons: set[str] = set()
            for t in list(spec.tendons):
                refs_joint = any(
                    p.type == mujoco.mjtObj.mjOBJ_JOINT and p.name == jname
                    for p in t.path
                )
                if refs_joint:
                    removed_tendons.add(t.name)
                    spec.delete(t)
            if removed_tendons:
                for e in list(spec.equalities):
                    if e.type == mujoco.mjtEq.mjEQ_TENDON and (
                        e.name1 in removed_tendons or e.name2 in removed_tendons
                    ):
                        spec.delete(e)
                for a in list(spec.actuators):
                    if (
                        a.trntype == mujoco.mjtTrn.mjTRN_TENDON
                        and a.target in removed_tendons
                    ):
                        spec.delete(a)
        if remove_actuators:
            for a in list(spec.actuators):
                if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == jname:
                    spec.delete(a)
        spec.delete(j)
    for child in list(parent.bodies):
        removed_joint_ids.extend(
            recursive_immobilize(
                spec, temp_model, child, remove_eqs, remove_actuators, remove_sites
            )
        )
    return removed_joint_ids


def recursive_remove_contacts(parent, return_condition=None):
    if return_condition is not None and return_condition(parent):
        return
    for g in parent.geoms:
        g.contype = 0
        g.conaffinity = 0
    for child in parent.bodies:
        recursive_remove_contacts(child, return_condition)


def recursive_mirror(meshes_to_mirror, spec_copy, parent):
    parent.pos[1] *= -1
    parent.quat[[1, 3]] *= -1
    parent.name += "_mirrored"
    for g in parent.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            spec_copy.delete(g)
            continue
        g.pos[1] *= -1
        g.quat[[1, 3]] *= -1
        g.name += "_mirrored"
        g.group = 1
        meshes_to_mirror.add(g.meshname)
        g.meshname += "_mirrored"
    for child in parent.bodies:
        if "ping_pong" in child.name:
            spec_copy.delete(child)
            continue
        recursive_mirror(meshes_to_mirror, spec_copy, child)
