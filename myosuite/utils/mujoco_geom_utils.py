# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Keep compiled-model fields consistent after runtime geometry edits."""

from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np

_PRIMITIVE_GEOM_TYPES = frozenset(
    int(t)
    for t in (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_BOX,
    )
)
_INERTIA_BOUND_OPTIONS = ("boundmass", "boundinertia", "balanceinertia")
_STATISTICS = ("meaninertia", "meanmass", "meansize", "extent", "center")


def refresh_geom_derived_fields(
    model: mujoco.MjModel, spec: mujoco.MjSpec, body_ids: Iterable[int]
) -> None:
    """Recompute the model fields MuJoCo derives from body geoms and body mass.

    Editing ``geom_type``, ``geom_size``, ``geom_pos``/``geom_quat`` or
    ``body_mass`` on a compiled model leaves stale every field the compiler
    derived from them: the bounding volumes that cull collisions
    (``geom_aabb``, ``geom_rbound``, the body BVH), the inertia and inertial
    frame of moving bodies, the ``*_sameframe`` kinematics flags, and the
    ``mj_setConst`` constants (``body_invweight0``, ``dof_invweight0``,
    ``body_subtreemass``, ...). Each listed body is compiled alone in a small
    spec, with its geoms taken from ``model``, their densities or masses from
    ``spec`` scaled so the body keeps its current ``body_mass``, and the
    compiler's inertia bounds; the results are copied back and
    ``mj_setConst`` is rerun on scratch data, keeping the statistics the model
    file sets. A moving body then matches a recompile with the same geometry
    and masses. A body welded to the world keeps its compiled inertial frame,
    which does not enter the dynamics.

    Explicit inertias retain their specified shape and scale with body mass.

    Args:
        model: Compiled model whose geoms or body masses were edited in place.
        spec: Spec that ``model`` was compiled from.
        body_ids: Ids of the bodies whose geoms or mass were edited.

    Raises:
        NotImplementedError: If a non-primitive geom has mass or collides,
            the body's set of colliding geoms changed, or the inertial frame of a
            body compiled as ``simple`` would move.
    """
    for body_id in body_ids:
        _refresh_body(model, spec, int(body_id))
    # mj_setConst recomputes every statistic; keep those the model file sets.
    user_stats = {
        name: np.copy(getattr(model.stat, name))
        for name in _STATISTICS
        if not np.isnan(np.ravel(getattr(spec.stat, name))[0])
    }
    mujoco.mj_setConst(model, mujoco.MjData(model))
    for name, value in user_stats.items():
        setattr(model.stat, name, value if value.ndim else float(value))


def _refresh_body(model: mujoco.MjModel, spec: mujoco.MjSpec, body_id: int) -> None:
    """Copy one body's derived fields from a standalone compile of the body."""
    name = model.body(body_id).name
    src_body = spec.bodies[body_id]
    static = model.body_weldid[body_id] == 0

    tiny = mujoco.MjSpec()
    tiny.compiler.inertiagrouprange = spec.compiler.inertiagrouprange
    tiny_body = tiny.worldbody.add_body()
    geom_ids, tiny_geoms = [], []
    for i, src in enumerate(src_body.geoms):
        gid = int(model.body_geomadr[body_id]) + i
        if int(model.geom_type[gid]) not in _PRIMITIVE_GEOM_TYPES:
            # A massless, non-colliding geom (e.g. a visual mesh) affects
            # neither the inertia nor the body BVH.
            if src.mass != 0 or model.geom_contype[gid] or model.geom_conaffinity[gid]:
                raise NotImplementedError(
                    f"Body {name!r} has a non-primitive geom with mass or contacts."
                )
            continue
        geom_ids.append(gid)
        tiny_geoms.append(
            tiny_body.add_geom(
                type=mujoco.mjtGeom(int(model.geom_type[gid])),
                size=model.geom_size[gid],
                pos=model.geom_pos[gid],
                quat=model.geom_quat[gid],
                group=int(model.geom_group[gid]),
                contype=int(model.geom_contype[gid]),
                conaffinity=int(model.geom_conaffinity[gid]),
                density=src.density,
                mass=src.mass,
                typeinertia=src.typeinertia,
            )
        )
    site_ids = np.flatnonzero(model.site_bodyid == body_id)
    for sid in site_ids:
        tiny_body.add_site(pos=model.site_pos[sid], quat=model.site_quat[sid])

    if static:
        # MuJoCo compiles static bodies as "simple", which pins their inertial
        # frame; the BVH and frame flags are expressed in that frame.
        tiny_body.explicitinertial = True
        tiny_body.mass = model.body_mass[body_id]
        tiny_body.ipos = model.body_ipos[body_id]
        tiny_body.iquat = model.body_iquat[body_id]
        tiny_body.inertia = model.body_inertia[body_id]
    else:
        if (
            spec.compiler.inertiafromgeom
            == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
            or (
                spec.compiler.inertiafromgeom
                == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
                and src_body.explicitinertial
            )
        ):
            tiny_body.explicitinertial = True
            tiny_body.mass = model.body_mass[body_id]
            tiny_body.ipos = model.body_ipos[body_id]
            scale = model.body_mass[body_id] / src_body.mass
            if np.isfinite(src_body.fullinertia[0]):
                tiny_body.fullinertia = src_body.fullinertia * scale
            else:
                tiny_body.iquat = model.body_iquat[body_id]
                tiny_body.inertia = src_body.inertia * scale
        else:
            # Scale geom masses so the body keeps its current mass.
            geom_mass = tiny.compile().body_mass[1]
            if geom_mass <= 0:
                raise NotImplementedError(f"Body {name!r} has no geom mass.")
            scale = model.body_mass[body_id] / geom_mass
            for geom in tiny_geoms:
                geom.density *= scale
                geom.mass *= scale
        for option in _INERTIA_BOUND_OPTIONS:
            setattr(tiny.compiler, option, getattr(spec.compiler, option))
    compiled = tiny.compile()

    adr, num = int(model.body_bvhadr[body_id]), int(model.body_bvhnum[body_id])
    if num != compiled.body_bvhnum[1]:
        raise NotImplementedError(f"Body {name!r} changed its set of colliding geoms.")
    if model.body_simple[body_id] and (
        compiled.body_sameframe[1] != mujoco.mjtSameFrame.mjSAMEFRAME_BODY
    ):
        raise NotImplementedError(
            f"Body {name!r} is compiled as simple, so its inertial frame cannot "
            "move; compile it with simple='false'."
        )

    model.geom_aabb[geom_ids] = compiled.geom_aabb
    model.geom_rbound[geom_ids] = compiled.geom_rbound
    model.geom_sameframe[geom_ids] = compiled.geom_sameframe
    model.site_sameframe[site_ids] = compiled.site_sameframe
    if not static:
        model.body_inertia[body_id] = compiled.body_inertia[1]
        model.body_ipos[body_id] = compiled.body_ipos[1]
        model.body_iquat[body_id] = compiled.body_iquat[1]
        model.body_sameframe[body_id] = compiled.body_sameframe[1]

    nodes = slice(adr, adr + num)
    model.bvh_aabb[nodes] = compiled.bvh_aabb
    model.bvh_child[nodes] = compiled.bvh_child
    model.bvh_depth[nodes] = compiled.bvh_depth
    # Leaf nodes store geom ids: map the standalone ids back to model ids.
    node_geom = compiled.bvh_nodeid.copy()
    leaf = node_geom >= 0
    node_geom[leaf] = np.asarray(geom_ids)[node_geom[leaf]]
    model.bvh_nodeid[nodes] = node_geom
