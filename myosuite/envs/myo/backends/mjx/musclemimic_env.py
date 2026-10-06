# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MJX MuscleMimic-compatible bimanual-arm task.

The implementation follows the same observation/action conventions used by
``amathislab/musclemimic`` for the MJX bimanual arm so policy checkpoints
can be replayed in MyoSuite with minimal translation.

Observation, reward, and task-sampling logic lives in
:class:`~myosuite.envs.myo.backends.mjx.musclemimic_base.MjxMuscleMimicBase`.
This class only implements :meth:`setup_model`.
"""

from __future__ import annotations

from ml_collections import config_dict
from mujoco import mjx

from myosuite.envs.myo.backends.mjx.musclemimic_base import MjxMuscleMimicBase
from myosuite.integrations.musclemimic.bimanual_model import (
    BODY2SITES_FOR_MIMIC as _BODY2SITES_FOR_MIMIC,
    compile_mimic_bimanual_mjmodel,
    default_mimic_config,
)


class MjxMuscleMimicIOEnv(MjxMuscleMimicBase):
    """MuscleMimic-style MJX bimanual-arm task.

    Args:
        config: Environment configuration dict (see
            :func:`~myosuite.integrations.musclemimic.bimanual_model.default_mimic_config`
        ).
        config_overrides: Optional key-value overrides applied on top of *config*.
    """

    def setup_model(self, config: config_dict.ConfigDict) -> None:
        """Compile bimanual MuJoCo/MJX model and init tracking indices.

        Args:
            config: Frozen environment config dict.
        """
        impl = getattr(config, "mjx_impl", None) or None
        self._mj_model, self._mj_spec, xml_path = compile_mimic_bimanual_mjmodel(config)
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._xml_path = xml_path
        self._n_substeps = int(config.ctrl_dt / config.sim_dt)
        self._init_tracking_state(config)


__all__ = [
    "MjxMuscleMimicIOEnv",
    "compile_mimic_bimanual_mjmodel",
    "default_mimic_config",
    "_BODY2SITES_FOR_MIMIC",
]
