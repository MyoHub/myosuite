# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""JAX muscle-fatigue model and FatigueWrapper for MJX environments.

Provides two public classes:

``CumulativeFatigue``
    JAX implementation of the 3CC-r muscle fatigue model.
    Adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701
    Based on implementation from Aleksi Ikkala and Florian Fischer.

``FatigueWrapper``
    Brax-style ``Wrapper`` that adds cumulative muscle fatigue dynamics to any
    ``MyoMjxEnvBase`` environment.  Fatigue state is stored in
    ``mjx.Data.userdata`` so that the wrapped step function remains fully
    JIT-compilable without Python-side state.

Ported from ``MyoHub/myosuite`` ``mjx`` branch (commit c22af6d).
"""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jp
import jax.random as jrandom
import mujoco
import numpy as np
from brax.envs.base import Wrapper
from ml_collections import config_dict, ConfigDict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from myosuite.envs.myo.backends.mjx.mjx_env_base import MyoMjxEnvBase
from myosuite.terms.base_action import sigmoid_muscle_activation

# Keys that may be included in the obs state from fatigue internals
ALLOWED_FATIGUE_OBS_KEYS: list[str] = ["MA", "MR", "MF"]

# Floating-point epsilon constants
_FLOAT_EPS: float = float(jp.finfo(jp.float32).eps)
_EPS4: float = _FLOAT_EPS * 4.0


class CumulativeFatigue:
    """3CC-r muscle fatigue model implemented in JAX.

    Tracks the fraction of active (MA), resting (MR), and fatigued (MF)
    motor units for each muscle actuator in the model.

    All state is returned as a ``dict[str, jax.Array]`` so that it can be
    stored in ``mjx.Data.userdata`` and remain compatible with JAX JIT.

    Args:
        mj_model: Compiled ``mujoco.MjModel``; used to count muscle actuators
            and read time constants.
        frame_skip: Number of physics substeps per control step, used to scale
            the integration timestep.
    """

    def __init__(self, mj_model: mujoco.MjModel, frame_skip: int = 1) -> None:
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.na: int = int(np.sum(muscle_act_ind))

        self.F = jp.array(0.00912, dtype=jp.float32)  # Fatigue coefficient
        self.R = jp.array(0.1 * 0.00094, dtype=jp.float32)  # Recovery coefficient
        self.r = jp.array(10 * 15, dtype=jp.float32)  # Recovery multiplier
        self.dt = jp.array(mj_model.opt.timestep * frame_skip, dtype=jp.float32)

        self.tauact = jp.array(
            [
                mj_model.actuator_dynprm[i][0]
                for i in range(len(muscle_act_ind))
                if muscle_act_ind[i]
            ],
            dtype=jp.float32,
        )
        self.taudeact = jp.array(
            [
                mj_model.actuator_dynprm[i][1]
                for i in range(len(muscle_act_ind))
                if muscle_act_ind[i]
            ],
            dtype=jp.float32,
        )

    def compute_act(
        self,
        TL: jax.Array,
        fatigue_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Advance fatigue state by one step given target activations *TL*.

        Args:
            TL: Target activation levels, shape ``(na,)``.
            fatigue_state: Dict with keys ``"MA"``, ``"MR"``, ``"MF"``.

        Returns:
            Updated fatigue state dict.
        """
        MA = fatigue_state["MA"]
        MR = fatigue_state["MR"]
        MF = fatigue_state["MF"]

        LD = 1 / self.tauact * (0.5 + 1.5 * MA)
        LR = (0.5 + 1.5 * MA) / self.taudeact

        C = jp.zeros_like(MA)
        mask1 = (MA < TL) & (MR > (TL - MA))
        C = jp.where(mask1, LD * (TL - MA), C)
        mask2 = (MA < TL) & (MR <= (TL - MA))
        C = jp.where(mask2, LD * MR, C)
        mask3 = MA >= TL
        C = jp.where(mask3, LR * (TL - MA), C)

        rR = jp.where(MA >= TL, self.r * self.R, self.R)

        C_min = jp.maximum(
            -MA / self.dt + self.F * MA,
            (MR - 1) / self.dt + rR * MF,
        )
        C_max = jp.minimum(
            (1 - MA) / self.dt + self.F * MA,
            MR / self.dt + rR * MF,
        )
        C = jp.clip(C, C_min, C_max)

        MA = MA + (C - self.F * MA) * self.dt
        MR = MR + (-C + rR * MF) * self.dt
        MF = MF + (self.F * MA - rR * MF) * self.dt

        return {"MA": MA, "MR": MR, "MF": MF}

    def get_effort(
        self,
        TL: jax.Array,
        fatigue_state: dict[str, jax.Array],
    ) -> jax.Array:
        """Compute effort as the norm of the difference between MA and TL.

        Args:
            TL: Target activation levels.
            fatigue_state: Dict with keys ``"MA"``, ``"MR"``, ``"MF"``.

        Returns:
            Scalar effort value.
        """
        return jp.linalg.norm(fatigue_state["MA"] - TL)

    def reset(
        self,
        rng: jax.Array,
        fatigue_reset_vec: list[float] | None = None,
        fatigue_reset_random: bool = False,
    ) -> dict[str, jax.Array]:
        """Sample an initial fatigue state.

        Args:
            rng: JAX random key (used only when *fatigue_reset_random* is True).
            fatigue_reset_vec: Optional per-muscle initial MF fractions.  When
                provided, ``MR = 1 - MF`` and ``MA = 0``.
            fatigue_reset_random: If True, sample random initial fractions.

        Returns:
            Dict with keys ``"MA"``, ``"MR"``, ``"MF"`` as float32 JAX arrays.

        Raises:
            AssertionError: If both *fatigue_reset_vec* and
                *fatigue_reset_random* are specified simultaneously, or if the
                length of *fatigue_reset_vec* does not match ``self.na``.
        """
        if fatigue_reset_random:
            assert (
                fatigue_reset_vec is None
            ), "Cannot use fatigue_reset_vec if fatigue_reset_random=True"
            key1, key2 = jrandom.split(rng)
            non_fatigued = jrandom.uniform(key1, (self.na,))
            active_frac = jrandom.uniform(key2, (self.na,))
            MA = non_fatigued * active_frac
            MR = non_fatigued * (1 - active_frac)
            MF = 1 - non_fatigued
        elif fatigue_reset_vec is not None:
            assert (
                len(fatigue_reset_vec) == self.na
            ), f"Invalid length of fatigue vector (expected {self.na}, got {len(fatigue_reset_vec)})"
            MF = jp.array(fatigue_reset_vec, dtype=jp.float32)
            MR = 1 - MF
            MA = jp.zeros(self.na, dtype=jp.float32)
        else:
            MA = jp.zeros(self.na, dtype=jp.float32)
            MR = jp.ones(self.na, dtype=jp.float32)
            MF = jp.zeros(self.na, dtype=jp.float32)

        return {"MA": MA, "MR": MR, "MF": MF}

    def set_FatigueCoefficient(self, F: float) -> None:
        """Set the fatigue coefficient *F*.

        Args:
            F: New fatigue coefficient value.
        """
        self.F = jp.array(F, dtype=jp.float32)

    def set_RecoveryCoefficient(self, R: float) -> None:
        """Set the recovery coefficient *R*.

        Args:
            R: New recovery coefficient value.
        """
        self.R = jp.array(R, dtype=jp.float32)

    def set_RecoveryMultiplier(self, r: float) -> None:
        """Set the recovery time multiplier *r*.

        Args:
            r: New multiplier value.
        """
        self.r = jp.array(r, dtype=jp.float32)


class FatigueWrapper(Wrapper):
    """Wrap a ``MyoMjxEnvBase`` env with cumulative muscle-fatigue dynamics.

    Fatigue state (MA, MR, MF) is stored in ``mjx.Data.userdata`` to remain
    fully JIT-compatible.  On each ``step()``, the wrapper:

    1. Applies the sigmoid action normalisation (since the inner env is
       configured with ``norm_actions=False`` to avoid double-normalisation).
    2. Updates the fatigue model with the normalised action.
    3. Replaces the muscle activations in the action with the active fraction MA.
    4. Passes the fatigued action to the inner env's step.
    5. Optionally appends fatigue state arrays (MA/MR/MF) to the observation.

    Args:
        env: A ``MyoMjxEnvBase`` instance to wrap.
        fatigue_config: ``ConfigDict`` with keys:
            ``fatigue_reset_vec`` (``None`` or list of per-muscle MF fractions),
            ``fatigue_reset_random`` (bool),
            ``fatigue_obs_keys`` (list of keys from ``ALLOWED_FATIGUE_OBS_KEYS``).
    """

    DEFAULT_MUSCLE_CONFIG: config_dict.ConfigDict = config_dict.create(
        fatigue_reset_vec=None,
        fatigue_reset_random=False,
        fatigue_obs_keys=[],
    )

    def __init__(
        self,
        env: MyoMjxEnvBase,
        fatigue_config: config_dict.ConfigDict = DEFAULT_MUSCLE_CONFIG,
    ) -> None:
        # Expand userdata to hold 3 × nu fatigue state values, then recompile
        self.nuserdata_without_fatigue: int = env.mj_model.nuserdata
        # Disable inner env normalisation — we apply it here before fatigue
        env._config.norm_actions = False
        env._mj_spec.nuserdata += env.mjx_model.nu * 3
        env._mj_model = env._mj_spec.compile()
        env._mjx_model = mjx.put_model(env._mj_model, impl=env.impl)

        super().__init__(env)

        self.fatigue_reset_vec = fatigue_config.fatigue_reset_vec
        self.fatigue_reset_random = fatigue_config.fatigue_reset_random
        self.fatigue_obs_keys: list[str] = list(fatigue_config.fatigue_obs_keys)
        assert all(k in ALLOWED_FATIGUE_OBS_KEYS for k in self.fatigue_obs_keys), (
            f"Invalid fatigue_obs_keys: {self.fatigue_obs_keys}. "
            f"Allowed keys are: {ALLOWED_FATIGUE_OBS_KEYS}"
        )

        self.muscle_act_ind: np.ndarray = (
            self.env.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        )
        self.muscle_fatigue = CumulativeFatigue(self.env.mj_model, self.n_substeps)

        _first = self.nuserdata_without_fatigue
        nu = self.env.mj_model.nu
        self.fatigue_index_MA = jp.arange(_first, _first + nu)
        self.fatigue_index_MR = jp.arange(_first + nu, _first + nu * 2)
        self.fatigue_index_MF = jp.arange(_first + nu * 2, _first + nu * 3)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment and initialise fatigue state.

        Args:
            rng: JAX random key.

        Returns:
            Initial ``State`` with fatigue stored in ``data.userdata``.
        """
        rng, rng_fati = jax.random.split(rng, 2)
        state = super().reset(rng)

        fatigue_state = self.muscle_fatigue.reset(
            rng=rng_fati,
            fatigue_reset_vec=self.fatigue_reset_vec,
            fatigue_reset_random=self.fatigue_reset_random,
        )
        new_userdata = state.data.userdata.at[self.fatigue_index_MA].set(
            fatigue_state["MA"]
        )
        new_userdata = new_userdata.at[self.fatigue_index_MR].set(fatigue_state["MR"])
        new_userdata = new_userdata.at[self.fatigue_index_MF].set(fatigue_state["MF"])
        state = state.replace(data=state.data.replace(userdata=new_userdata))
        state = state.replace(obs=self._add_fatigue_to_obs(state.obs, state.data))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Advance one step with fatigue dynamics applied.

        Args:
            state: Current ``State`` (fatigue in ``data.userdata``).
            action: Raw policy action.

        Returns:
            Next ``State``.
        """
        norm_action = sigmoid_muscle_activation(action, jp)

        prev_fatigue = {
            "MA": state.data.userdata[self.fatigue_index_MA],
            "MR": state.data.userdata[self.fatigue_index_MR],
            "MF": state.data.userdata[self.fatigue_index_MF],
        }
        fatigue_state = self.muscle_fatigue.compute_act(
            norm_action[self.muscle_act_ind], fatigue_state=prev_fatigue
        )

        new_userdata = state.data.userdata.at[self.fatigue_index_MA].set(
            fatigue_state["MA"]
        )
        new_userdata = new_userdata.at[self.fatigue_index_MR].set(fatigue_state["MR"])
        new_userdata = new_userdata.at[self.fatigue_index_MF].set(fatigue_state["MF"])

        # Replace desired activations with currently active motor-unit fraction
        action_fatigued = norm_action.at[self.muscle_act_ind].set(fatigue_state["MA"])

        state = state.replace(data=state.data.replace(userdata=new_userdata))
        next_state = super().step(state, action_fatigued)
        next_state = next_state.replace(
            obs=self._add_fatigue_to_obs(next_state.obs, next_state.data)
        )
        return next_state

    def _add_fatigue_to_obs(
        self, obs: dict[str, Any], data: mjx.Data
    ) -> dict[str, Any]:
        """Append requested fatigue arrays to the ``"state"`` observation.

        Args:
            obs: Current obs dict (must contain ``"state"`` key).
            data: Current ``mjx.Data`` with fatigue in ``userdata``.

        Returns:
            Updated obs dict.
        """
        if "state" not in obs:
            return obs
        obs_state = obs["state"]
        if "MA" in self.fatigue_obs_keys:
            obs_state = jp.concatenate(
                [obs_state, data.userdata[self.fatigue_index_MA]], axis=-1
            )
        if "MR" in self.fatigue_obs_keys:
            obs_state = jp.concatenate(
                [obs_state, data.userdata[self.fatigue_index_MR]], axis=-1
            )
        if "MF" in self.fatigue_obs_keys:
            obs_state = jp.concatenate(
                [obs_state, data.userdata[self.fatigue_index_MF]], axis=-1
            )
        return {**obs, "state": obs_state}

    def set_fatigue_reset_random(self, fatigue_reset_random: bool) -> None:
        """Toggle random fatigue initialisation.

        Args:
            fatigue_reset_random: If True, use random initial fatigue state.
        """
        self.fatigue_reset_random = fatigue_reset_random

    @classmethod
    def skim_config(
        cls,
        config: ConfigDict,
        config_overrides: dict[str, Any] | None = None,
    ) -> tuple[ConfigDict, ConfigDict]:
        """Extract fatigue-specific keys from *config* and *config_overrides*.

        Separates ``fatigue_config`` sub-dict from the main env config so that
        the base env constructor doesn't receive unknown keys.

        Args:
            config: Full environment config that may contain a ``fatigue_config``
                sub-dict.
            config_overrides: Optional override dict; fatigue keys are extracted
                and applied to *fatigue_config*.

        Returns:
            Tuple ``(config, fatigue_config)`` where *config* no longer contains
            ``fatigue_config`` and *fatigue_config* holds the merged values.
        """
        if config_overrides is None:
            config_overrides = {}
        # Deep-copy the class default to avoid mutating it across calls.
        fatigue_config = copy.deepcopy(FatigueWrapper.DEFAULT_MUSCLE_CONFIG)
        if "fatigue_config" in config:
            fatigue_config = config.fatigue_config
            del config.fatigue_config
        for k in list(fatigue_config.keys()):
            if k in config_overrides:
                fatigue_config[k] = config_overrides.pop(k)
        config.update(config_overrides)
        return config, fatigue_config
