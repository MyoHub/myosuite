# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

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


def delete_tendon(spec: mujoco.MjSpec, tendon: mujoco.MjsTendon) -> None:
    """Delete *tendon* together with the actuators, equalities and sensors on it.

    Args:
        spec: Spec that owns *tendon*.
        tendon: Tendon to delete.
    """
    name = tendon.name
    for a in list(spec.actuators):
        if a.trntype == mujoco.mjtTrn.mjTRN_TENDON and a.target == name:
            spec.delete(a)
    for e in list(spec.equalities):
        if e.type == mujoco.mjtEq.mjEQ_TENDON and name in (e.name1, e.name2):
            spec.delete(e)
    for s in list(spec.sensors):
        if s.objtype == mujoco.mjtObj.mjOBJ_TENDON and s.objname == name:
            spec.delete(s)
    spec.delete(tendon)


def delete_joint(spec: mujoco.MjSpec, joint: mujoco.MjsJoint) -> None:
    """Delete *joint* together with every element that references it.

    Removes the fixed tendons that use the joint (with their own dependents,
    see :func:`delete_tendon`), joint equalities, joint actuators and joint
    sensors, so the spec still compiles afterwards.

    Args:
        spec: Spec that owns *joint*.
        joint: Joint to delete.
    """
    name = joint.name
    for t in list(spec.tendons):
        if any(
            w.type == mujoco.mjtWrap.mjWRAP_JOINT
            and getattr(w.target, "name", None) == name
            for w in t.path
        ):
            delete_tendon(spec, t)
    for e in list(spec.equalities):
        if e.type == mujoco.mjtEq.mjEQ_JOINT and name in (e.name1, e.name2):
            spec.delete(e)
    for a in list(spec.actuators):
        if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == name:
            spec.delete(a)
    for s in list(spec.sensors):
        if s.objtype == mujoco.mjtObj.mjOBJ_JOINT and s.objname == name:
            spec.delete(s)
    spec.delete(joint)


def recursive_immobilize(
    spec,
    temp_model,
    parent,
    remove_eqs=False,
    remove_actuators=False,
    remove_sites=True,
    remove_dependents=False,
):
    """Delete every joint in the subtree of *parent*.

    Args:
        spec: Spec to edit.
        temp_model: Model compiled from *spec* before the edit (for qpos addresses).
        parent: Root body of the subtree.
        remove_eqs: Also delete joint equalities on the removed joints.
        remove_actuators: Also delete joint actuators on the removed joints.
        remove_sites: Also delete the sites of the subtree.
        remove_dependents: Delete each joint with :func:`delete_joint`, i.e. with
            all fixed tendons, equalities, actuators and sensors that use it.

    Returns:
        qpos addresses of the removed joints.
    """
    removed_joint_ids = []
    if remove_sites:
        for s in parent.sites:
            spec.delete(s)
    for j in parent.joints:
        removed_joint_ids.extend(temp_model.joint(j.name).qposadr)
        if remove_dependents:
            delete_joint(spec, j)
            continue
        if remove_eqs:
            for e in spec.equalities:
                if e.type == mujoco.mjtEq.mjEQ_JOINT and (
                    e.name1 == j.name or e.name2 == j.name
                ):
                    spec.delete(e)
        if remove_actuators:
            for a in spec.actuators:
                if a.trntype == mujoco.mjtTrn.mjTRN_JOINT and a.target == j.name:
                    spec.delete(a)
        spec.delete(j)
    for child in parent.bodies:
        removed_joint_ids.extend(
            recursive_immobilize(
                spec,
                temp_model,
                child,
                remove_eqs,
                remove_actuators,
                remove_sites,
                remove_dependents,
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
