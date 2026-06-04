# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MJX MuscleMimic-compatible full-body task (MyoFullBody).

Uses
:func:`~myosuite.integrations.musclemimic.fullbody_model.compile_mimic_fullbody_mjmodel`
and the same observation/reward layout as
:class:`~myosuite.envs.myo.backends.mjx.musclemimic_env.MjxMuscleMimicIOEnv` so
policies can be trained or rolled out entirely through MyoSuite (``make`` +
JAX) without the MuscleMimic sandbox checkout.

Observation, reward, and task-sampling logic lives in
:class:`~myosuite.envs.myo.backends.mjx.musclemimic_base.MjxMuscleMimicBase`.
This class only implements :meth:`setup_model`.

Pretrained MuscleMimic *locomotion* checkpoints expect the upstream LocoMuJoCo
observation stack; they will not match this env's observation vector unless you
retrain or add a compatible adapter.
"""

from __future__ import annotations

from ml_collections import config_dict
from mujoco import mjx

from myosuite.envs.myo.backends.mjx.musclemimic_base import MjxMuscleMimicBase
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
)


class MjxMuscleMimicFullbodyEnv(MjxMuscleMimicBase):
    """MuscleMimic-style MJX full-body task.

    Args:
        config: Environment configuration dict (see
            :func:`~myosuite.integrations.musclemimic.fullbody_model.default_mimic_fullbody_config`
        ).
        config_overrides: Optional key-value overrides applied on top of *config*.
    """

    def setup_model(self, config: config_dict.ConfigDict) -> None:
        """Compile full-body MuJoCo/MJX model and init tracking indices.

        Applies solver tuning knobs (``model_iterations``,
        ``model_ls_iterations``, ``model_disableflags``) from *config* after
        compilation; the CPU-only compile function leaves these at MuJoCo
        defaults.

        Args:
            config: Frozen environment config dict.
        """
        impl = getattr(config, "mjx_impl", None) or None
        self._mj_model, self._mj_spec, xml_path = compile_mimic_fullbody_mjmodel(config)
        self._mj_model.opt.iterations = int(config.model_iterations)
        self._mj_model.opt.ls_iterations = int(config.model_ls_iterations)
        self._mj_model.opt.disableflags = int(config.model_disableflags)

        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._xml_path = xml_path
        self._n_substeps = int(config.ctrl_dt / config.sim_dt)
        self._init_tracking_state(config)


__all__ = ["MjxMuscleMimicFullbodyEnv"]
