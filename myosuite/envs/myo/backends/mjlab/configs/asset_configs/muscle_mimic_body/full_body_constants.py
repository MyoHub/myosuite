# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from ml_collections import config_dict

from myosuite.integrations.musclemimic import build_mimic_fullbody_spec

_ALL_XML_ACTUATORS = EntityArticulationInfoCfg(
    actuators=(
        XmlActuatorCfg(
            target_names_expr=(".*_tendon$",), transmission_type=TransmissionType.TENDON
        ),
    )
)


def _spec_only(*args, **kwargs):
    # The mimic fullbody spec builder also returns the xml of the body, but we need only the spec.
    spec = build_mimic_fullbody_spec(*args, **kwargs)[0]
    spec.add_sensor(
        name="lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="head_mimic",
    )
    spec.add_sensor(
        name="ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="head_mimic",
    )
    return spec


def get_full_body_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=partial(_spec_only, config=config_dict.create(disable_fingers=True)),
        articulation=_ALL_XML_ACTUATORS,
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    body = Entity(get_full_body_cfg())
    model = body.spec.compile()
    from mjviser import Viewer

    data = mujoco.MjData(model)
    Viewer(model, data).run()
