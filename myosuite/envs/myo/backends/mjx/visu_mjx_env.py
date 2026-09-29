# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Visualize a trained PPO policy on any MyoSuite MJX env (saved by train_jax_ppo.py)."""

import argparse
import json
import os
import pickle
from types import SimpleNamespace

import jax
import mujoco
import mujoco.viewer
import numpy as np
from brax.envs.wrappers.training import EpisodeWrapper
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from loop_rate_limiters import RateLimiter

from myosuite.envs.myo.backends.mjx import make, ppo_config

WINDOW = 100  # number of reward samples shown in the plot


def main(args):
  run_dir = os.path.join(args.logdir, args.env_name)

  # Always use the config saved at training time.
  with open(os.path.join(run_dir, "config.pickle"), "rb") as f:
    saved_config = pickle.load(f)

  env = make(args.env_name, config_overrides=saved_config)
  env = EpisodeWrapper(env, env._config.max_episode_steps, 1)
  m, d = env.mj_model, mujoco.MjData(env.mj_model)

  # Note: the first two steps are slow because of jitting.
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step, donate_argnums=(0,))

  rng, key = jax.random.split(jax.random.PRNGKey(0))
  ctx = SimpleNamespace(
    state=jit_reset(key), rng=rng, reset_requested=False,
    running_reward=0.0, episode_reward=0.0,
    time_hist=np.zeros(WINDOW), reward_hist=np.zeros(WINDOW),
  )
  policy = get_policy(env, ctx.state.obs, run_dir)

  def step_fn(m, d):
    actions = policy(ctx.state.obs)
    ctx.state = jit_step(ctx.state, actions)
    st = ctx.state

    d.ctrl, d.qpos, d.act, done = jax.device_get(
      (st.data.ctrl, st.data.qpos, st.data.act, st.done))
    mujoco.mj_forward(m, d)

    # Color tendons by muscle activation (skipped if act doesn't map 1:1 to tendons).
    if d.act.shape[0] == m.ntendon:
      c = np.sqrt(np.sqrt(d.act))[:, None]
      m.tendon_rgba = c * np.array([0.95, 0.3, 0.3, 1]) + (1 - c) * np.array([0.05, 0.05, 0.05, 1])

    reward = float(st.reward)
    ctx.running_reward += reward
    ctx.time_hist = np.roll(ctx.time_hist, -1)
    ctx.reward_hist = np.roll(ctx.reward_hist, -1)
    ctx.time_hist[-1] = float(st.data.time)
    ctx.reward_hist[-1] = reward

    if done or ctx.reset_requested:
      ctx.rng, key = jax.random.split(ctx.rng)
      ctx.state = jit_reset(key)
      ctx.reset_requested = False
      ctx.episode_reward, ctx.running_reward = ctx.running_reward, 0.0
      ctx.time_hist, ctx.reward_hist = np.zeros(WINDOW), np.zeros(WINDOW)
      print("Reset")

  if os.name == "posix":
    run_viser(m, d, step_fn, ctx)
  else:
    lrl = RateLimiter(1 / env._config.ctrl_dt)
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
      viewer.sync()
      while viewer.is_running():
        step_fn(m, d)
        viewer.sync()
        lrl.sleep()


def run_viser(m, d, step_fn, ctx):
  import viser
  from mjviser import Viewer

  ui = SimpleNamespace(ready=False)

  def render_fn(scene):
    scene.update_from_mjdata(d)

    if not ui.ready:
      gui = scene.server.gui
      with gui.add_folder("Reward"):
        ui.reward = gui.add_text("Step reward", "0.0", disabled=True)
        ui.episode = gui.add_text("Last episode reward", "0.0", disabled=True)
        ui.plot = scene.server.add_uplot(
          data=(ctx.time_hist, ctx.reward_hist),
          series=(viser.uplot.Series(label="Time (s)"),
                  viser.uplot.Series(label="Reward", stroke="#ff6384", width=2)),
          title="Reward Timeline", aspect=1.5)
      ui.ready = True

    ui.reward.value = f"{ctx.reward_hist[-1]:.4f}"
    ui.episode.value = f"{ctx.episode_reward:.4f}"
    ui.plot.data = (ctx.time_hist, ctx.reward_hist)

    # Mirror the policy's controls onto the actuator sliders (without triggering callbacks).
    for slider, _ in ui.viewer._actuator_sliders:
      slider._impl.update_cb = []
    for slider, act_id in ui.viewer._actuator_sliders:
      slider.value = round(float(np.clip(d.ctrl[act_id], slider.min, slider.max)), 3)

  def reset_fn(m, d):
    ctx.reset_requested = True

  ui.viewer = Viewer(m, d, step_fn=step_fn, render_fn=render_fn, reset_fn=reset_fn)
  ui.viewer.run()


def get_policy(env, obs_example, run_dir):
  """Rebuild the PPO network as train_jax_ppo.py does and load the saved params."""
  network_cfg = dict(ppo_config).get("network_factory", {})
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape, obs_example)
  ppo_network = ppo_networks.make_ppo_networks(
    obs_shape, env.action_size, preprocess_observations_fn=normalize, **network_cfg)

  with open(os.path.join(run_dir, "playground_params.pickle"), "rb") as f:
    params = pickle.load(f)
  norm_p, pol_p = ((params["normalizer_params"], params["policy_params"])
                   if isinstance(params, dict) else (params[0], params[1]))

  @jax.jit
  def deterministic_policy(obs):
    logits = ppo_network.policy_network.apply(norm_p, pol_p, obs)
    return ppo_network.parametric_action_distribution.mode(logits)

  return deterministic_policy


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Visualize a trained MJX policy")
  parser.add_argument("--env_name", type=str, default="MjxElbowPoseRandom-v0",
                      help="Must match the --env_name used in training.")
  parser.add_argument("--logdir", type=str, default="./mjx_logs",
                      help="Root dir containing <env_name>/{playground_params.pickle, config.pickle}.")
  main(parser.parse_args())
