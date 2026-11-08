# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoSuite's full body (myo_sim ``myofullbody``) as an mjlab entity for the multi-clip tracking task.

The body is myo_sim's ``myofullbody`` as the ChaseTag full-body envs build it
(:func:`~myosuite.integrations.musclemimic.fullbody_model.build_native_mimic_fullbody_spec`: fingers disabled,
31 independent joint DoF, 354 muscles), with the edits muscle-supermimic's ``myofullbody_cfg.py`` makes for its task
(github.com/kaibohex/muscle-supermimic @ 8a6244b, Apache-2.0): no floor, light or keyframe (mjlab's scene owns the
ground), the root turned to face +x, and only the four foot touch sensors kept. The mimic sites go too (the task
tracks their bodies, each site sits at its body's origin). ``model_name`` picks another myo_sim full body (e.g. a
later ``myofullbody`` release) with the same body and joint names. The muscles' [-1, 1] ``ctrlrange`` of MyoSuite's
build changes nothing, since MuJoCo clamps a muscle's control to [0, 1] and the task's excitation action is in [0, 1].
"""

from __future__ import annotations

import functools

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from ml_collections import config_dict

BODY_MODEL = "myofullbody"
"""The myo_sim model of the body (the release MyoSuite pins)."""
ROOT_BODY = "Full Body"
"""The body of the free joint, the frame that holds the pelvis (welded to it) and the spine."""
FREE_JOINT = "root"
INIT_POS: tuple[float, float, float] = (0.0, 0.0, 1.0)
INIT_ROT: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)
"""+90 deg about z: the default pose faces +x, the canonical root frame's forward axis."""
MUSCLE_TENDON_EXPR: tuple[str, ...] = (r".*_tendon(_l|_left)?",)
"""Every muscle tendon (myo_sim names left-arm tendons ``*_tendon_l``, musclemimic_models ``*_tendon_left``)."""
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis", "lumbar1", "head",
    "humerus_l", "ulna_l", "lunate_l",
    "humerus_r", "ulna_r", "lunate_r",
    "femur_l", "tibia_l", "talus_l", "toes_l",
    "femur_r", "tibia_r", "talus_r", "toes_r",
)  # fmt: skip
"""The bodies of MuscleMimic's 17 mimic sites (each at its body's origin), pelvis first."""
FOOT_BODIES: tuple[str, ...] = ("calcn_r", "toes_r", "calcn_l", "toes_l")
"""The bodies of the foot touch sensor sites, in MuscleMimic's order."""


def body_spec(
    self_collisions: bool = True, model_name: str = BODY_MODEL
) -> mujoco.MjSpec:
    """The task's full-body spec (fingers disabled).

    Args:
        self_collisions: Keep the explicit self-contact pairs (MuscleMimic keeps them).
        model_name: The myo_sim full-body model.

    Returns:
        A new, uncompiled spec.
    """
    # imported here: the MyoSuite env registry imports this module while registering the task
    from myosuite.integrations.musclemimic.fullbody_model import (  # noqa: PLC0415
        FULLBODY_BODY2SITES_FOR_MIMIC,
        build_native_mimic_fullbody_spec,
    )

    spec = build_native_mimic_fullbody_spec(
        config_dict.create(disable_fingers=True, model_name=model_name)
    )
    spec.njmax = spec.nconmax = spec.nuser_jnt = -1
    spec.nkey = 0
    spec.option.timestep = mujoco.MjSpec().option.timestep
    for key in list(spec.keys):
        spec.delete(key)
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    assert spec.joints[0].name == FREE_JOINT
    root = spec.body(ROOT_BODY)
    root.pos, root.quat = INIT_POS, INIT_ROT
    for site in list(spec.sites):
        if site.name in FULLBODY_BODY2SITES_FOR_MIMIC.values():
            spec.delete(site)
    for sensor in list(spec.sensors):
        if sensor.type != mujoco.mjtSensor.mjSENS_TOUCH:
            spec.delete(sensor)
    if not self_collisions:
        for pair in list(spec.pairs):
            spec.delete(pair)
    return spec


def body_cfg(self_collisions: bool = True, model_name: str = BODY_MODEL) -> EntityCfg:
    """Free-floating full body; a muscle's ``ctrl`` is its excitation, written as its tendon's effort target.

    Args:
        self_collisions: See :func:`body_spec`.
        model_name: See :func:`body_spec`.

    Returns:
        The entity config.
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS, rot=INIT_ROT, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        spec_fn=functools.partial(
            body_spec, self_collisions=self_collisions, model_name=model_name
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=MUSCLE_TENDON_EXPR,
                    transmission_type=TransmissionType.TENDON,
                    command_field="effort",
                ),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


@functools.lru_cache(maxsize=2)
def compiled_body(model_name: str = BODY_MODEL) -> mujoco.MjModel:
    """The entity compiled on its own (no scene, no name prefix), for CPU-side kinematics and tests."""
    return body_spec(model_name=model_name).compile()
