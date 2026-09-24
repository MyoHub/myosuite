# Modified from code in Mujoco Playground to suit this project
#
#
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrappers for MuJoCo Playground environments."""

import functools
from absl import logging
from typing import Any, Callable, Optional, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers.training import VmapWrapper, DomainRandomizationVmapWrapper
from brax.base import System
from brax import base

from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper, BraxDomainRandomizationVmapWrapper, BraxAutoResetWrapper

# class BraxDomainRandomizationVmapWrapper(Wrapper):
#   """Brax wrapper for domain randomization."""

#   def __init__(
#       self,
#       env: mjx_env.MjxEnv,
#       randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
#   ):
#     super().__init__(env)
#     self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

#   def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
#     env = self.env
#     env.unwrapped._mjx_model = mjx_model
#     return env

#   def reset(self, rng: jax.Array) -> mjx_env.State:
#     def reset(mjx_model, rng):
#       env = self._env_fn(mjx_model=mjx_model)
#       return env.reset(rng)

#     state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
#     return state

#   def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
#     def step(mjx_model, s, a):
#       env = self._env_fn(mjx_model=mjx_model)
#       return env.step(s, a)

#     res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
#         self._mjx_model_v, state, action
#     )
#     return res


def _identity_vision_randomization_fn(
    mjx_model: mjx.Model, num_worlds: int
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary fields for the Madrona memory buffer copy."""
  in_axes = jax.tree_util.tree_map(lambda x: None, mjx_model)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_type': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })
  mjx_model = mjx_model.tree_replace({
      'geom_rgba': jp.repeat(
          jp.expand_dims(mjx_model.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jp.repeat(
          jp.expand_dims(mjx_model.geom_matid, 0), num_worlds, axis=0
      ),
      'geom_size': jp.repeat(
          jp.expand_dims(mjx_model.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jp.repeat(
          jp.expand_dims(mjx_model.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jp.repeat(
          jp.expand_dims(mjx_model.light_dir, 0), num_worlds, axis=0
      ),
      'light_type': jp.repeat(
          jp.expand_dims(mjx_model.light_type, 0), num_worlds, axis=0
      ),
      'light_castshadow': jp.repeat(
          jp.expand_dims(mjx_model.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jp.repeat(
          jp.expand_dims(mjx_model.light_cutoff, 0), num_worlds, axis=0
      ),
  })
  return mjx_model, in_axes


def _supplement_vision_randomization_fn(
    mjx_model: mjx.Model,
    randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    num_worlds: int,
) -> Tuple[mjx.Model, mjx.Model]:
  """Tile the necessary missing fields for the Madrona memory buffer copy."""
  mjx_model, in_axes = randomization_fn(mjx_model)

  required_fields = [
      'geom_rgba',
      'geom_matid',
      'geom_size',
      'light_pos',
      'light_dir',
      'light_type',
      'light_castshadow',
      'light_cutoff',
  ]

  for field in required_fields:
    if getattr(in_axes, field) is None:
      in_axes = in_axes.tree_replace({field: 0})
      val = getattr(mjx_model, field)
      mjx_model = mjx_model.tree_replace({
          field: jp.repeat(jp.expand_dims(val, 0), num_worlds, axis=0),
      })
  return mjx_model, in_axes


class MadronaWrapper:
  """Wraps a MuJoCo Playground to be used in Brax with Madrona."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      num_worlds: int,
      randomization_fn: Optional[
          Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
      ] = None,
  ):
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_vision_randomization_fn, num_worlds=num_worlds
      )
    else:
      randomization_fn = functools.partial(
          _supplement_vision_randomization_fn,
          randomization_fn=randomization_fn,
          num_worlds=num_worlds,
      )
    self._env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    self.num_worlds = num_worlds

    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_type',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    return self._env.reset(rng)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics."""
    return self._env.step(state, action)

  def __getattr__(self, name):
    """Delegate attribute access to the wrapped instance."""
    return getattr(self._env.unwrapped, name)

class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat
    try: 
      non_accumulation_metrics = env.non_accumulation_metrics
      print(f"Found non-accumulation metrics: {non_accumulation_metrics}")
    except AttributeError:
      print("No non-accumulation metrics found")
      non_accumulation_metrics = []
    self.non_accumulation_metrics = non_accumulation_metrics

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if (metric_name != 'reward') and (metric_name not in self.non_accumulation_metrics):
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
      if metric_name in self.non_accumulation_metrics:
        print(f"Setting non-accumulation metric: {metric_name} to the current value")
        state.info['episode_metrics'][metric_name] = state.metrics[metric_name]
    state.info['episode_done'] = done
    return state.replace(done=done)


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate, should
      equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over
    full_reset: whether to call `env.reset` during `env.step` on done rather
      than resetting to a cached first state. Setting full_reset=True may
      increase wallclock time because it forces full resets to random states.

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if vision:
    env = MadronaWrapper(env, num_worlds=num_vision_envs, randomization_fn=randomization_fn)
  elif randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = EpisodeWrapper(env, episode_length, action_repeat)
  env = BraxAutoResetWrapper(env, full_reset=full_reset)

  return env

def wrap_myosuite_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """

    if vision:
      env = MadronaWrapper(env, num_worlds=num_vision_envs, randomization_fn=randomization_fn)
    elif randomization_fn is None:
      env = VmapWrapper(env)
    else:
      env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)

    env = EpisodeWrapper(env, episode_length, action_repeat)

    # Select Auto-Reset Wrapper
    if vision:
      env = AutoResetWrapper(env)
    else:
      env = BetterAutoResetWrapper(env)

    return env


class BetterAutoResetWrapper(Wrapper):
    """Automatically "resets" Brax envs that are done, without clearing info state."""
    
    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        
        def return_state_as_it_is(current_state):
          return current_state

        def reset_state(current_state):
          new_state = self.env.auto_reset(current_state)
          new_state = new_state.replace(done=current_state.done, metrics=current_state.metrics)
          keys = set(current_state.info.keys()) - set(new_state.info.keys())
          for k in keys:
            new_state.info[k] = current_state.info[k]
          return new_state


        def compute_next_state(current_state):
          return jax.lax.cond(current_state.done, reset_state, return_state_as_it_is, current_state)

        vmapped_compute_next_state = jax.vmap(compute_next_state)
        next_state = vmapped_compute_next_state(state)
        return next_state

class AutoResetWrapper(Wrapper):
    """Automatically "resets" Brax envs that are done, without clearing info state."""

    ## AutoReset Wrapper required to implement adaptive target curriculum; checks if episode is completed and calls reset inside this function;
    ## WARNING: Due to the following lines, applying the default Brax AutoResetWrapper has no effect to this env!

    def step(self, state: mjx_env.State, action: jax.Array):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        # ####################################################################################

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        state_after_reset = jax.vmap(self.env.auto_reset)(state)
        # state_after_reset = self.env.reset(rng)  #, state.info)

        for k in state.info:
            state_after_reset.info[k] = state_after_reset.info.get(k, state.info[k])

        data = jax.tree.map(
            where_done, state_after_reset.data, state.data  # state.pipeline_state
        )
        # obs_dict = jax.tree.map(where_done, obs_dict_after_reset, obs_dict)
        obs = jax.tree.map(where_done, state_after_reset.obs, state.obs)  # state.obs)
        info = jax.tree.map(where_done, state_after_reset.info, state.info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=state.reward,
            done=state.done,
            metrics=state.metrics,
            info=info,
        )

class EvalVmapWrapper(Wrapper):
  """Vectorizes Brax env for evaluation runs, using eval_reset instead of reset as entrypoint."""

  def __init__(self, env: Env, n_randomizations: Optional[int] = None, predefined_evals: Optional[bool] = True):
    super().__init__(env)
    # self.batch_size = batch_size
    self.n_randomizations = n_randomizations
    self.predefined_evals = predefined_evals
    self.eval_wrapped = True

  def eval_reset(self, rng: jax.Array, eval_id: jax.Array) -> State:
    # TODO: make sure that this runs with n_episodes=1
    batch_size = len(eval_id)
    if batch_size > 1:
      rng = jax.random.split(rng, batch_size)
    if self.predefined_evals:
      if self.n_randomizations is not None:
        ## only allow for eval_id values between 0 and (self.n_randomizations - 1) in this case
        eval_id = eval_id % self.n_randomizations
      return jax.vmap(self.env.eval_reset)(rng, eval_id)
    else:
      return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.step)(state, action)
  

def _maybe_wrap_env_for_evaluation(eval_env, seed):
#     if eval_env.vision:
#       return eval_env, None
    rng = jax.random.PRNGKey(seed)
    # predefined_evals = True
    # try:
    #   n_randomizations = eval_env.prepare_eval_rollout(rng)
    # except AssertionError:
    #   print("ERROR: No evaluations defined for this task! Will use random tunnels for evaluation instead...")
    n_randomizations = 1
    predefined_evals = False
    
    if not hasattr(eval_env, "eval_wrapped"):
      eval_env = EvalVmapWrapper(eval_env, n_randomizations=n_randomizations, predefined_evals=predefined_evals)
      assert hasattr(eval_env, "eval_wrapped")
    
    return eval_env, n_randomizations