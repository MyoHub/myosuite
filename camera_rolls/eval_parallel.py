import jax
import mujoco
import numpy as np
import os
import imageio

from src.control_learning.eval.multi_view import duplicate
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from mujoco_playground import registry

# Replace with your network architecture
ppo_network = ppo_networks.make_ppo_networks(
  130,
  39,
  policy_hidden_layer_sizes=[128, 128, 128, 64],
  preprocess_observations_fn=normalize,
  policy_obs_key='hand_obs')

dirname = os.path.dirname(__file__)

# Load your own params
model_path = '../../learned_models/hand_policy/closedloop_params.pickle'
model_path = os.path.join(dirname, model_path)
hl_params = model.load_params(model_path)
hl_policy = ppo_networks.make_inference_fn(ppo_network)(hl_params)


# You might want to use a deterministic policy
@jax.jit
def deterministic_hl_policy(input_data):
  logits = ppo_network.policy_network.apply(hl_params['normalizer_params'], hl_params['policy_params'], input_data)
  brax_result = ppo_network.parametric_action_distribution.mode(logits)
  return brax_result


def get_env(ctrl_dt):
  return registry.load('MyoHand', config_overrides={"ctrl_dt": ctrl_dt})  # Replace with your env!


def main():
  ctrl_dt = 0.008
  env = get_env(ctrl_dt)
  m, d = get_mj_model_data(env)

  mujoco.mj_forward(m, d)

  # Camera work params
  wait_time = 10.0
  interp_duration = 10.0
  total_duration = wait_time + interp_duration + 5.0

  # This body will be aimed at at the start (the prefix comes from the duplicate() function
  target_body = "inst_7_0-hamate"
  start_offset = np.array([0.4, 0, 0])
  body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_body)
  start_lookat = d.xpos[body_id].copy() if body_id != -1 else np.zeros(3)
  start_pos = start_lookat + start_offset

  # Currently we just aim ait a global position
  end_lookat = np.array([0, 0, 2])
  end_offset = np.array([8.0, -8.0, 4.0])
  end_pos = end_lookat + end_offset

  def get_cam_params(pos, lookat):
    vec = np.array(pos) - np.array(lookat)
    dist = np.linalg.norm(vec)

    if dist < 1e-6:
      return 0.0, 0.0, 0.0, np.array(lookat)
    azimuth_rad = np.arctan2(-vec[1], -vec[0])
    elevation_rad = -np.arcsin(vec[2] / dist)
    azimuth_deg = np.degrees(azimuth_rad)
    elevation_deg = np.degrees(elevation_rad)
    return float(dist), float(azimuth_deg), float(elevation_deg), np.array(lookat)

  s_dist, s_az, s_el, s_look = get_cam_params(start_pos, start_lookat)
  e_dist, e_az, e_el, e_look = get_cam_params(end_pos, end_lookat)

  def ease_in_out(t):
    return t * t * (3 - 2 * t)

  jit_reset = jax.jit(jax.vmap(env.reset))
  jit_step = jax.jit(jax.vmap(env.step))
  key = jax.random.PRNGKey(0)
  state = jit_reset(jax.random.split(key, 64))
  deterministic = True
  actions = deterministic_hl_policy({'hand_obs': state.obs['hand_obs']})
  smoothing = 0.2

  width, height = 1280, 720
  renderer = mujoco.Renderer(m, height=height, width=width, max_geom=30000)
  renderer.scene.flags[5] = 1

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(m, cam)
  fps = int(1.0 / ctrl_dt / 2)  # Subsample frames
  video_path = 'simulation_output.mp4'
  writer = imageio.get_writer(video_path, fps=fps)

  print(f"Starting headless render. Video will be saved to: {video_path}")

  elapsed = 0.0
  i = 0  # for subsampling

  while elapsed < total_duration:
    hl_key, ll_key, key = jax.random.split(key, 3)

    actions = ((smoothing * actions) + ((1 - smoothing) * deterministic_hl_policy(state.obs))
               if deterministic
               else (smoothing * actions) + ((1 - smoothing)
                                             * hl_policy(state.obs, hl_key)))
    actions = np.clip(actions, 0, 2)
    state = jit_step(state, actions)
    d.qpos = np.concatenate(state.data.qpos)
    mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.

    if i % 2 == 0:
      elapsed = state.data.time[0]

      if elapsed < wait_time:
        p_dist, p_az, p_el, p_look = s_dist, s_az, s_el, s_look
      else:
        t = np.clip((elapsed - wait_time) / interp_duration, 0.0, 1.0)
        t_eased = ease_in_out(t)

        p_dist = s_dist + (e_dist - s_dist) * t_eased
        p_az = s_az + (e_az - s_az) * t_eased
        p_el = s_el + (e_el - s_el) * t_eased
        p_look = s_look + (e_look - s_look) * t_eased

      cam.distance = p_dist
      cam.azimuth = p_az
      cam.elevation = p_el
      cam.lookat[:] = p_look
      renderer.update_scene(d, camera=cam)
      frame = renderer.render()
      writer.append_data(frame)
    i += 1

    if int(elapsed / 0.016) % 100 == 0:
      print(f"Rendered: {elapsed:.2f}s / {total_duration:.2f}s")

  writer.close()
  print(f"Video saved as {video_path}")


def get_mj_model_data(env):
  spec = mujoco.MjSpec.from_file(env.xml_path)
  spec = env.preprocess_spec(spec)
  spec = duplicate(spec)
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 720
  m = spec.compile()
  d = mujoco.MjData(m)
  return m, d


if __name__ == '__main__':
  main()
