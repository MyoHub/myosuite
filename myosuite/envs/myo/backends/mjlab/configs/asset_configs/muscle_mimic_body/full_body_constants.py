# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU MuJoCo model builder for MuscleMimic-compatible MyoFullBody.

Mirrors ``MuscleMimic`` ``MyoFullBody._apply_spec_changes`` (finger removal,
mimic sites, muscle ``ctrlrange``) for the ``musclemimic_models`` full-body
MJCF — importable without MJX or ``mujoco_playground``.

Does **not** apply ``_modify_spec_for_mjx`` (MJX contact stripping / warp
budgets); that path is only used when building the JAX/Warp training env.

Finger joint/muscle name lists are shared with
:mod:`myosuite.integrations.musclemimic.bimanual_model` (same names as upstream
``MyoFullBody``).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from functools import partial

from etils import epath
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from ml_collections import config_dict

from myosuite.integrations.musclemimic import build_mimic_fullbody_spec

_ALL_XML_ACTUATORS = EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(".*_tendon$",),
                                                                         transmission_type=TransmissionType.TENDON),))

def _spec_only(*args, **kwargs):
  # The mimic fullbody spec builder also returns the xml of the body, but we need only the spec.
  return build_mimic_fullbody_spec(*args, **kwargs)[0]

def get_full_body_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=partial(_spec_only, config=config_dict.create(disable_fingers=True)),
    articulation = _ALL_XML_ACTUATORS
  )


if __name__ == "__main__":
  from mjlab.entity.entity import Entity
  body = Entity(get_full_body_cfg())
  model = body.spec.compile()
  from mjviser import Viewer
  data = mujoco.MjData(model)
  Viewer(model, data).run()

