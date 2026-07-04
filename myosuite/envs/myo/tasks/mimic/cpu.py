# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU Gymnasium environments for MuscleMimic bimanual/full-body tasks."""

from __future__ import annotations

from typing import Any

# pylint: disable=no-member
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.integrations.musclemimic.bimanual_model import (
    BODY2SITES_FOR_MIMIC,
    compile_mimic_bimanual_mjmodel,
    default_mimic_config,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.terms.mimic_obs import (
    resolve_mimic_site_ids,
    sample_mimic_target_sites,
)
from myosuite.terms.mimic_reward import (
    MimicTrackingConfig,
    compute_mimic_reward,
    compute_mimic_tracking_error,
)


class _MuscleMimicCpuBase(MyoGymnasiumEnv, EzPickle):
    """Shared CPU implementation for MuscleMimic task variants."""

    def __init__(
        self,
        frame_skip: int,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        low = kwargs.get("mimic_target_low")
        high = kwargs.get("mimic_target_high")
        if low is None or high is None:
            raise ValueError(
                "mimic_target_low and mimic_target_high must be supplied "
                "(see MjxMuscleMimicBase.sample_task box sampling)."
            )
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        EzPickle.__init__(self, frame_skip, seed, **kwargs)
        self._site_ids: np.ndarray | None = None
        self._target_site_pos: np.ndarray | None = None
        self._target_lo = np.asarray(low, dtype=np.float64)
        self._target_hi = np.asarray(high, dtype=np.float64)
        self._tracking_cfg = MimicTrackingConfig()

    def _setup_spaces(self) -> None:
        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl[:, 0],
            high=ctrl[:, 1],
            dtype=np.float32,
        )
        obs_size = int(self.model.nq + self.model.nv + self.model.na + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def _resolve_mimic_sites(self, names: tuple[str, ...]) -> np.ndarray:
        return resolve_mimic_site_ids(self.model, names)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        qpos = accessor.joint_pos().astype(np.float32)
        qvel = accessor.joint_vel().astype(np.float32)
        act = accessor.muscle_act().astype(np.float32)
        assert self._site_ids is not None
        assert self._target_site_pos is not None
        track_err = compute_mimic_tracking_error(
            self.data.site_xpos[self._site_ids],
            self._target_site_pos,
        )
        return {
            "qpos": qpos,
            "qvel": qvel,
            "act": act,
            "track_err": np.asarray([track_err], dtype=np.float32),
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        track_err = float(obs_dict["track_err"][0])
        solved = bool(track_err < self._tracking_cfg.success_threshold)
        dense = compute_mimic_reward(track_err, self._tracking_cfg)
        return {
            "track_err": track_err,
            "dense": dense,
            "sparse": -track_err,
            "solved": solved,
            "done": False,
        }

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        assert self._site_ids is not None
        self._target_site_pos = sample_mimic_target_sites(
            np_random,
            self._target_lo,
            self._target_hi,
            int(self._site_ids.shape[0]),
        )
        return {}


class MuscleMimicBimanualEnv(_MuscleMimicCpuBase):
    """CPU Gymnasium MuscleMimic bimanual task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        cfg = default_mimic_config()
        mt_low = kwargs.pop(
            "mimic_target_low",
            tuple(float(x) for x in cfg.target_site_range.low),
        )
        mt_high = kwargs.pop(
            "mimic_target_high",
            tuple(float(x) for x in cfg.target_site_range.high),
        )
        super().__init__(
            frame_skip=frame_skip,
            seed=seed,
            mimic_target_low=mt_low,
            mimic_target_high=mt_high,
            **kwargs,
        )
        self.model, self._mj_spec, self._xml_path = compile_mimic_bimanual_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        self._site_ids = self._resolve_mimic_sites(tuple(BODY2SITES_FOR_MIMIC.values()))
        self._target_site_pos = np.zeros(
            (int(self._site_ids.shape[0]), 3), dtype=np.float32
        )
        self._setup_spaces()


class MuscleMimicFullbodyEnv(_MuscleMimicCpuBase):
    """CPU Gymnasium MuscleMimic full-body task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        cfg = default_mimic_fullbody_config()
        mt_low = kwargs.pop(
            "mimic_target_low",
            tuple(float(x) for x in cfg.target_site_range.low),
        )
        mt_high = kwargs.pop(
            "mimic_target_high",
            tuple(float(x) for x in cfg.target_site_range.high),
        )
        super().__init__(
            frame_skip=frame_skip,
            seed=seed,
            mimic_target_low=mt_low,
            mimic_target_high=mt_high,
            **kwargs,
        )
        self.model, self._mj_spec, self._xml_path = compile_mimic_fullbody_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        names = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())
        self._site_ids = self._resolve_mimic_sites(names)
        self._target_site_pos = np.zeros(
            (int(self._site_ids.shape[0]), 3), dtype=np.float32
        )
        self._setup_spaces()
