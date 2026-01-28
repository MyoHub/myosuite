import jax
import jax.numpy as jp
import jax.random as jrandom
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
from typing import Dict, Tuple, Any
from brax.envs.base import Wrapper
import numpy as np
from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase
from ml_collections import config_dict, ConfigDict

ALLOWED_FATIGUE_OBS_KEYS = ["MA", "MR", "MF"]

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0

class CumulativeFatigue:
    """
    JAX implementation of the 3CC-r muscle fatigue model
    Adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701
    Based on implementation from Aleksi Ikkala and Florian Fischer
    """

    def __init__(self, mj_model, frame_skip=1):
        # Get muscle actuator indices
        muscle_act_ind = mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Convert to concrete integer value
        self.na = int(np.sum(muscle_act_ind))  # Use numpy here since we're in __init__

        # Parameters (will be included in tree_flatten children)
        self.F = jp.array(0.00912, dtype=jp.float32)  # Fatigue coefficient
        self.R = jp.array(0.1 * 0.00094, dtype=jp.float32)  # Recovery coefficient
        self.r = jp.array(10 * 15, dtype=jp.float32)  # Recovery multiplier
        self.dt = jp.array(mj_model.opt.timestep * frame_skip, dtype=jp.float32)

        # Get muscle parameters
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

    # @jax.jit
    def compute_act(self, TL, fatigue_state):
        """
        Compute muscle activation considering fatigue

        Args:
            TL: Target activation levels

        Returns:
            fatigue_state: dict with vectors containing the ratio of active/resting/fatigued muscle 
                        units for each muscle (keys are "MA", "MR", and "MF")
        """

        MA = fatigue_state["MA"]
        MR = fatigue_state["MR"]
        MF= fatigue_state["MF"]

        # Calculate effective time constants
        LD = 1 / self.tauact * (0.5 + 1.5 * MA)
        LR = (0.5 + 1.5 * MA) / self.taudeact

        # Calculate C(t) - transfer rate between MR and MA
        C = jp.zeros_like(MA)

        # Case 1: MA < TL and MR > (TL - MA)
        mask1 = (MA < TL) & (MR > (TL - MA))
        C = jp.where(mask1, LD * (TL - MA), C)

        # Case 2: MA < TL and MR <= (TL - MA)
        mask2 = (MA < TL) & (MR <= (TL - MA))
        C = jp.where(mask2, LD * MR, C)

        # Case 3: MA >= TL
        mask3 = MA >= TL
        C = jp.where(mask3, LR * (TL - MA), C)

        # Calculate recovery rate
        rR = jp.where(MA >= TL, self.r * self.R, self.R)

        # Clip C(t) to ensure states remain between 0 and 1
        C_min = jp.maximum(
            -MA / self.dt + self.F * MA,
            (MR - 1) / self.dt + rR * MF,
        )
        C_max = jp.minimum(
            (1 - MA) / self.dt + self.F * MA, MR / self.dt + rR * MF
        )
        C = jp.clip(C, C_min, C_max)

        # Update states
        dMA = (C - self.F * MA) * self.dt
        dMR = (-C + rR * MF) * self.dt
        dMF = (self.F * MA - rR * MF) * self.dt

        MA += dMA
        MR += dMR
        MF += dMF

        fatigue_state = {"MA": MA,
                         "MR": MR,
                         "MF": MF}

        return fatigue_state

    # @jax.jit
    def get_effort(self, TL, fatigue_state):
        """Calculate effort as norm of difference between actual and target activation"""
        MA = fatigue_state["MA"]
        return jp.linalg.norm(MA - TL)

    def reset(self, rng, fatigue_reset_vec=None, fatigue_reset_random=False):
        """Reset fatigue state.
        
        State attributes:
        - MA: Percentage of active muscle units (vector of length self.na)
        - MR: Percentage of resting muscle units (vector of length self.na)
        - MF: Percentage of fatigued muscle units (vector of length self.na)
        """
        if fatigue_reset_random:
            assert (
                fatigue_reset_vec is None
            ), "Cannot use fatigue_reset_vec if fatigue_reset_random=True"
            key1, key2 = jrandom.split(rng)
            non_fatigued_muscles = jrandom.uniform(key1, (self.na,))
            active_percentage = jrandom.uniform(key2, (self.na,))
            MA = non_fatigued_muscles * active_percentage
            MR = non_fatigued_muscles * (1 - active_percentage)
            MF = 1 - non_fatigued_muscles
        else:
            if fatigue_reset_vec is not None:
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

        fatigue_state = {"MA": MA,
                         "MR": MR,
                         "MF": MF}

        return fatigue_state

    def set_FatigueCoefficient(self, F):
        """Set Fatigue coefficient"""
        self.F = jp.array(F, dtype=jp.float32)

    def set_RecoveryCoefficient(self, R):
        """Set Recovery coefficient"""
        self.R = jp.array(R, dtype=jp.float32)

    def set_RecoveryMultiplier(self, r):
        """Set Recovery time multiplier"""
        self.r = jp.array(r, dtype=jp.float32)


class FatigueWrapper(Wrapper):
  """Wrapper that adds a CumulativeFatigue instance to the environment."""

  DEFAULT_MUSCLE_CONFIG = config_dict.create(fatigue_reset_vec= None,
                                             fatigue_reset_random=False,
                                             fatigue_obs_keys= [])

  def __init__(self, env: MjxMyoBase, fatigue_config=DEFAULT_MUSCLE_CONFIG):
    ## Increase nuserdata and recompile model
    self.nuserdata_without_fatigue = env.mj_model.nuserdata

    env._mj_spec.nuserdata += env.mjx_model.nu * 3
    env._mj_model = env._mj_spec.compile()
    env._mjx_model = mjx.put_model(env._mj_model, impl=env.impl)

    super().__init__(env)
    
    self.fatigue_reset_vec = fatigue_config.fatigue_reset_vec
    self.fatigue_reset_random = fatigue_config.fatigue_reset_random
    self.fatigue_obs_keys = fatigue_config.fatigue_obs_keys
    assert all([key in ALLOWED_FATIGUE_OBS_KEYS for key in self.fatigue_obs_keys]), \
        f"Invalid fatigue_obs_keys: {self.fatigue_obs_keys}. Allowed keys are: {ALLOWED_FATIGUE_OBS_KEYS}"
    # self.sex = muscle_config.sex
    # self.control_type = muscle_config.control_type
    # self.muscle_noise_params = muscle_config.noise_params

    self.muscle_act_ind = self.env.mj_model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE


    self.muscle_fatigue = CumulativeFatigue(
        self.env.mj_model, self.n_substeps
    )
    _fatigue_index_first = self.nuserdata_without_fatigue
    nu = self.env.mj_model.nu
    self.fatigue_index_MA = jp.arange(_fatigue_index_first, _fatigue_index_first + nu * 1)
    self.fatigue_index_MR = jp.arange(_fatigue_index_first + nu * 1, _fatigue_index_first + nu * 2)
    self.fatigue_index_MF = jp.arange(_fatigue_index_first + nu * 2, _fatigue_index_first + nu * 3)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, rng_fati = jax.random.split(rng, 2)

    state = super().reset(rng)
  
    fatigue_state = self.muscle_fatigue.reset(
        fatigue_reset_vec=self.fatigue_reset_vec,
        fatigue_reset_random=self.fatigue_reset_random,
        rng=rng_fati,
    )
    new_userdata = state.data.userdata.at[self.fatigue_index_MA].set(fatigue_state["MA"])
    new_userdata = new_userdata.at[self.fatigue_index_MR].set(fatigue_state["MR"])
    new_userdata = new_userdata.at[self.fatigue_index_MF].set(fatigue_state["MF"])
    data = state.data.replace(userdata=new_userdata)
    state = state.replace(data=data)

    ## add fatigue state to observation, if respective config keys are specified
    state = state.replace(
        obs=self.add_fatigue_to_obs(
            state.obs, state.data
        )
    )

    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    norm_action = 1.0/(1.0+jp.exp(-5.0*(action-0.5))) 
    previous_fatigue_state = {}
    previous_fatigue_state["MA"] = state.data.userdata[self.fatigue_index_MA]
    previous_fatigue_state["MR"] = state.data.userdata[self.fatigue_index_MR]
    previous_fatigue_state["MF"] = state.data.userdata[self.fatigue_index_MF]

    ## update fatigue state
    fatigue_state = self.muscle_fatigue.compute_act(norm_action[self.muscle_act_ind], fatigue_state=previous_fatigue_state)

    new_userdata = state.data.userdata.at[self.fatigue_index_MA].set(fatigue_state["MA"])
    new_userdata = new_userdata.at[self.fatigue_index_MR].set(fatigue_state["MR"])
    new_userdata = new_userdata.at[self.fatigue_index_MF].set(fatigue_state["MF"])

    ## replace desired activations with currently active motor units
    norm_action = norm_action.at[self.muscle_act_ind].set(fatigue_state["MA"])

    ## undo previous normalisation, as it is reapplied by super().step method
    action_fatigued = jp.log(1.0/norm_action - 1.0)/(-5.0) + 0.5

    ## store fatigue params in userdata
    data = state.data.replace(userdata=new_userdata)
    state = state.replace(data=data)

    ## perform main simulation step
    next_state = super().step(state, action_fatigued)

    ## add fatigue state to observation, if respective config keys are specified
    next_state = next_state.replace(
        obs=self.add_fatigue_to_obs(
            next_state.obs, next_state.data
        )
    )

    return next_state
  
  def add_fatigue_to_obs(
    self, obs: dict, data: mjx.Data) -> dict:
    """Observe qpos, qvel, act and qpos_err."""
    if "state" not in obs:
       return obs
    obs_state = obs["state"]
    if "MA" in self.fatigue_obs_keys:
      obs_state = jp.concatenate([obs_state, data.userdata[self.fatigue_index_MA]], axis=-1)
    if "MR" in self.fatigue_obs_keys:
      obs_state = jp.concatenate([obs_state, data.userdata[self.fatigue_index_MR]], axis=-1)
    if "MF" in self.fatigue_obs_keys:
      obs_state = jp.concatenate([obs_state, data.userdata[self.fatigue_index_MF]], axis=-1)
    return {**obs, **{"state": obs_state}}

  def set_fatigue_reset_random(self, fatigue_reset_random):
    self.fatigue_reset_random = fatigue_reset_random

  @classmethod
  def skim_config(cls, config: ConfigDict, config_overrides=None):
    if config_overrides is None:
      config_overrides = {}
    fatigue_config = FatigueWrapper.DEFAULT_MUSCLE_CONFIG
    if "fatigue_config" in config:
      fatigue_config = config.fatigue_config
      del config.fatigue_config
    for k in fatigue_config.keys():
      if k in config_overrides:
        fatigue_config[k] = config_overrides[k]
        del config_overrides[k]
    config.update(config_overrides)
    return config, fatigue_config
