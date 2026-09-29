# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bolt sprinter ABI freeze — must stay bit-compatible with Athletic API.

Surfaces locked for plant=myosprinter, task=sprint:

* Env IDs: ``myoSprint-v0``, ``Mjlab-Sprint-Flat-BoltMjWarp``
* FastTD3 obs/act: 336 / 138 (Bolt-convention packing in td3_adapter when present)
* Physics: ``BoltMuscleAction`` + HC Warp via ``make_sprint_env_cfg``
* Factory: ``mjlab_env_cfg_from_bundle(sprint, myosprinter)`` ≡ ``make_sprint_env_cfg``

Golden tests: ``test_myo_sprint_mjlab_parity.py``, ``test_myo_sprint_bolt_transfer.py``,
``scripts/msk_envs_repro/mjlab_investigation/verify_bitexact_multienv.py``.
"""

from __future__ import annotations

BOLT_SPRINTER_ENV_ID = "myoSprint-v0"
BOLT_SPRINTER_ALIAS_IDS = ("Mjlab-Sprint-Flat-BoltMjWarp",)
BOLT_SPRINTER_ENTITY = "sprinter"
BOLT_SPRINTER_HEADING_DIR = (1.0, 0.0)
# make_sprint_env_cfg FALL_HEIGHT (Y-up); do not drift to AthleticSpec default 0.6.
BOLT_SPRINTER_FALL_HEIGHT = 0.55
BOLT_SPRINTER_CTRL_DT = 1.0 / 30.0
BOLT_SPRINTER_SIM_DT = 1e-4
BOLT_SPRINTER_DECIMATION = 333
BOLT_SPRINTER_LAMBDA_VEL = 1e-2
BOLT_SPRINTER_LAMBDA_LIMIT = -3e-4
