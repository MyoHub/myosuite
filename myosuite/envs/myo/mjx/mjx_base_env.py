# Base class for universal functionality for MSk environments in MJX/Warp. This is at the moment separated from
# the MyoSuite classic sim abstraction, due to fundamental differences on how learning/simulation state is handled
# in previous learning frameworks and the highly parallelized MJX/Warp. However, I don't see this as fully
# irreconcilable. Currently we are building on top of MuJoCo playground's base env.
# TODO: Consider if the two implementations could be merged.

import logging
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src
from abc import ABC, abstractmethod
import numpy as np


class MjxMyoBase(mjx_env.MjxEnv, ABC):
    def __init__(
            self,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        self.impl = self._config.impl
        spec = self.preprocess_spec(spec)
        self._mj_spec = spec
        self._mj_model = spec.compile()

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = config.model_path.as_posix()

        self._n_substeps = int(config.ctrl_dt / config.sim_dt)

    def preprocess_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        for geom in spec.geoms:
            if self.impl == "jax":
                if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    geom.conaffinity = 0
                    geom.contype = 0
                    print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
                if geom.type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD) and geom.margin != 0:
                    geom.margin = 0
                    print(f"Margin of \"{geom.name}\" set to 0")
        spec.option.iterations = 6  # TODO: Parametrize with defaults in config?
        spec.option.ls_iterations = 6
        spec.option.ccd_iterations = 75 
        spec.option.timestep = self._config.sim_dt
        print(f"Iterations: {spec.option.iterations}, LS Iterations: {spec.option.ls_iterations}")
        #  TODO: consider which disableflags (self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP) and solver is
        #        most appropriate for the base preprocess. (mujoco.mjtSolver.mjSOL_NEWTON perhaps?)
        return spec

    @classmethod
    def norm_actions(cls, action):
        return 1.0 / (1.0 + jp.exp(-5.0 * (action - 0.5)))

    @abstractmethod
    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state. Abstract, so child class needs to provide actual implementation"""

        info = {'rng': rng,
                'step_count': jp.array(0, dtype=jp.int32)}  # These are mandatory fields needed
        obs = {}
        metrics = {}
        data = self._get_data(jp.zeros(self._mj_model.nq), jp.zeros(self._mj_model.nv))

        return State(data, obs, 0., 0., metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        state = self._step_simulation(state, action)
        # Performed in multiple steps as the stages commonly depend on each other in this direction. HLO is expected to
        # correct for inefficiency.
        state = state.replace(obs=self._get_obs(state.data, state.info))
        state = state.replace(reward=self._get_reward(state.data, state.info))
        state = state.replace(done=self._get_done(state))
        state = state.replace(
            metrics={**state.metrics, **self._get_metrics(state)})  # Other metrics get added by learning
        state = state.replace(info=self._get_info(state))
        return state

    def _step_simulation(self, state, action):
        norm_action = self.__class__.norm_actions(action)
        return state.replace(data=mjx_env.step(self.mjx_model, state.data, norm_action, self._n_substeps),
                             info={**state.info, "step_count": state.info["step_count"] + 1}
                            )

    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        """Must return a state with the observations replaced with the updated dict."""
        obs = jp.concatenate([
            data.qpos,
            data.qvel * self.mjx_model.opt.timestep,
            data.act,
        ])
        return {"base_obs": obs}

    def _get_reward(self, data: mjx.Data, info: dict) -> float:
        """Return a scalar value."""
        return sum(jax.tree_util.tree_leaves(self._get_rewards(data, info)))

    @abstractmethod
    def _get_rewards(self, data: mjx.Data, info: dict) -> dict:
        """Return a dictionary of rewards."""
        return {}

    def _get_done(self, state: State) -> float:
        """Return 1 for done"""
        return 0.

    def _get_metrics(self, state: State) -> dict:
        return {}

    def _get_info(self, state: State) -> dict:
        info = state.info
        return info

    def _get_data(self, qpos, qvel):
        naconmax = 50 * self._config.num_envs
        data = make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros((self.mjx_model.nu,)),
            impl=self._config.impl,
            naconmax=naconmax,
            njmax=self._mj_model.njmax if self._mj_model.njmax != -1 else 1_000,
            naccdmax=naconmax,  # https://github.com/google-deepmind/mujoco/pull/3096
        )
        return data

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def make_data(
        model: mujoco.MjModel,
        qpos: Optional[jax.Array] = None,
        qvel: Optional[jax.Array] = None,
        ctrl: Optional[jax.Array] = None,
        act: Optional[jax.Array] = None,
        mocap_pos: Optional[jax.Array] = None,
        mocap_quat: Optional[jax.Array] = None,
        impl: Optional[str] = None,
        naconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        naccdmax: Optional[int] = None,
        device: Optional[jax.Device] = None,
) -> mjx.Data:
    """Initialize MJX Data."""
    data = mjx.make_data(
        model,
        impl=impl,
        naconmax=naconmax,
        njmax=njmax,
        naccdmax=naccdmax,
        device=device,
    )
    if qpos is not None:
        data = data.replace(qpos=qpos)
    if qvel is not None:
        data = data.replace(qvel=qvel)
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)
    if act is not None:
        data = data.replace(act=act)
    if mocap_pos is not None:
        data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
    if mocap_quat is not None:
        data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
    return data
