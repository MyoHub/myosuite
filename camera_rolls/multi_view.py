import copy
from typing import Tuple

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

# import src.control_learning.envs
from myosuite.envs.myo.mjx import get_default_config, make as mjx_make
from mujoco_playground._src import registry


def _offset_tree(body, offset: Tuple[float, float, float]):
  body.pos = [
    body.pos[0] + offset[0],
    body.pos[1] + offset[1],
    body.pos[2] + offset[2],
  ]


def replicate_with_attach(parent_spec, child_spec, grid=(3, 3), spacing=(2, 2)):
  rows, cols = grid
  dx, dy = spacing
  parent_spec.copy_during_attach = True
  mesh_set = set()
  for r in range(rows):
    for c in range(cols):
      prefix = f"inst_{r}_{c}-"

      frame = parent_spec.worldbody.add_frame(
        pos=[c * dx - grid[0] * dx / 2, r * dy - grid[1] * dy / 2, 0.0],
      )
      # frame.quat = R.from_euler(seq='xyz', angles=np.array([0, 0, np.random.uniform(0, 2 * np.pi)])).as_quat(
      #   scalar_first=True)

      parent_spec.attach(
        child_spec,
        frame=frame,
        prefix=prefix,
      )

      for g in parent_spec.geoms:
        g.meshname = g.meshname.split("-")[-1]

      for m in parent_spec.meshes:
        if m.name.split("-")[-1] in mesh_set and "-" in m.name:
          parent_spec.delete(m)
          continue
        m.name = m.name.split("-")[-1]
        mesh_set.add(m.name)

  return parent_spec


def add_environment(spec: mujoco.MjSpec):
  tex = spec.add_texture(
    name="chequered",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    width=512,
    height=512,
    rgb1=[0.9, 0.9, 0.9],
    rgb2=[0.7, 0.7, 0.7],
  )

  sky = spec.add_texture(
    name="sky",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
    width=8,
    height=8,
    rgb1=[1, 1, 1],
  )

  mat = spec.add_material(
    name='grid', texrepeat=[50, 50], reflectance=.05
  ).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'chequered'

  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[50, 50, 0.1],
    material='grid',
  )

  for l in spec.lights:
    spec.delete(l)

  spec.worldbody.add_light(
    pos=[0, 0, 10],
    dir=[0.1, 0.2, -1],
    diffuse=[0.8, 0.8, 0.8],
    specular=[0.2, 0.2, 0.2],
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
  )

  spec.visual.map.fogstart = 0.3
  spec.visual.map.fogend = 0.8
  spec.visual.rgba.fog[:] = 1
  # spec.visual.rgba.haze[:] = 1

  return spec


def get_env(env_name="MjxElbowPoseRandom-v0", impl="warp"):
  # return registry.load('MyoHand', )
  return mjx_make(env_name, config_overrides={"impl": impl})  # Overwrite with your env's name
  # config = get_default_config(env_name)


def duplicate(base_spec):
  parent = mujoco.MjSpec()
  child = base_spec

  spec = replicate_with_attach(parent, child, grid=(8, 8))
  spec = add_environment(spec)
  return spec


if __name__ == "__main__":
  env = get_env()
  spec = mujoco.MjSpec.from_file(env.xml_path)
  spec = env.preprocess_spec(spec)

  parent = mujoco.MjSpec()
  child = spec

  spec = replicate_with_attach(parent, child, grid=(8, 8))
  spec = add_environment(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.user_scn.flags[5] = 1
    viewer.sync()
    while viewer.is_running():
      mujoco.mj_kinematics(model, data)
      viewer.sync()
