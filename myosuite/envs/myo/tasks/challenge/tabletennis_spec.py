# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared MuJoCo MjSpec preprocessing for MyoChallenge TableTennis (CPU + mjlab)."""

from __future__ import annotations

import mujoco

from myosuite.utils.spec_processing import (
    recursive_immobilize,
    recursive_mirror,
    recursive_remove_contacts,
)


def preprocess_tabletennis_spec(
    spec: mujoco.MjSpec,
    *,
    remove_body_collisions: bool = True,
    add_left_arm: bool = True,
) -> mujoco.MjSpec:
    """Apply the TableTennis challenge MJCF transforms used by ``TableTennisEnv``.

    Mirrors the historical ``TableTennisEnv._preprocess_spec`` logic so CPU
    (``ModelBuilder``) and mjlab (``EntityCfg.spec_fn``) compile the same scene.

    Args:
        spec: Root ``MjSpec`` loaded from ``myoarm_tabletennis.xml`` (or equivalent).
        remove_body_collisions: When True, strip some body-body collisions under
            ``full_body`` (same as Gymnasium env default).
        add_left_arm: When True, mirror and attach the left arm subtree.

    Returns:
        The same ``spec`` instance after in-place mutation.
    """
    for body in spec.bodies:
        if "paddle" in body.name and body.parent != spec.worldbody:
            break
    for s in spec.sensors:
        if "pingpong" not in s.name and "paddle" not in s.name:
            spec.delete(s)
    temp_model = spec.compile()
    removed_ids = recursive_immobilize(
        spec, temp_model, spec.body("femur_l"), remove_eqs=True
    )
    removed_ids.extend(
        recursive_immobilize(spec, temp_model, spec.body("femur_r"), remove_eqs=True)
    )
    for key in spec.keys:
        key.qpos = [j for i, j in enumerate(key.qpos) if i not in removed_ids]
    if remove_body_collisions:
        recursive_remove_contacts(
            spec.body("full_body"), return_condition=lambda b: "radius" in b.name
        )
    if add_left_arm:
        torso = spec.body("torso")
        spec_copy: mujoco.MjSpec = spec.copy()
        attachment_frame = torso.add_frame(
            quat=[0.5, 0.5, -0.5, 0.5], pos=[0.05, 0.373, -0.04]
        )
        for col in (
            spec_copy.keys,
            spec_copy.textures,
            spec_copy.materials,
            spec_copy.tendons,
            spec_copy.actuators,
            spec_copy.equalities,
            spec_copy.sensors,
            spec_copy.cameras,
        ):
            for item in list(col):
                spec_copy.delete(item)
        recursive_immobilize(spec_copy, temp_model, spec_copy.worldbody)
        recursive_remove_contacts(spec_copy.worldbody, return_condition=None)
        meshes_to_mirror: set[str] = set()
        recursive_mirror(meshes_to_mirror, spec_copy, spec_copy.body("clavicle"))
        for mesh in list(spec_copy.meshes):
            if mesh.name in meshes_to_mirror:
                mesh.name += "_mirrored"
                mesh.scale[1] *= -1
            else:
                spec_copy.delete(mesh)
        attachment_frame.attach_body(spec_copy.body("clavicle_mirrored"))
        spec.body("ulna_mirrored").quat = [0.546, 0, 0, -0.838]
        spec.body("humerus_mirrored").quat = [0.924, 0.383, 0, 0]
    return spec
