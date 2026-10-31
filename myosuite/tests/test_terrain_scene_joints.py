"""Extra scene free bodies must not change TERRA's trained observation layout."""

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace
from myosuite.integrations.musclemimic.fullbody_local_policy import FullbodyStateAdapter
from myosuite.integrations.musclemimic.terrain_observation import (
    TerrainObservation,
    TerrainObsCfg,
)

pytestmark = pytest.mark.tier1


def test_terrain_ignores_joints_outside_actor_subtree() -> None:
    """Keep movable scene object state outside the trained TERRA actor layout."""
    actor = """<body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size=".1" mass="1"/><site name="pelvis_mimic"/>
      <body pos="0 0 -.2"><joint name="hip"/>
        <geom type="sphere" size=".08" mass="1"/>
      </body>
    </body>"""
    pumpkin = """<body name="pumpkin" pos="3 0 2">
      <freejoint name="pumpkin_free"/><geom type="sphere" size=".1" mass="1"/>
    </body>"""
    base = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + actor + "</worldbody></mujoco>"
    )
    scene = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + actor + pumpkin + "</worldbody></mujoco>"
    )
    base_data, scene_data = mujoco.MjData(base), mujoco.MjData(scene)
    mujoco.mj_forward(base, base_data)
    mujoco.mj_forward(scene, scene_data)
    expected = FullbodyStateAdapter(base, egocentric_root=True).build_state(base_data)
    # Generic full-body observations retain their previous all-joint behavior.
    assert (
        FullbodyStateAdapter(scene, egocentric_root=True).build_state(scene_data).size
        == expected.size + 13
    )
    cfg = TerrainObsCfg(sites=("pelvis_mimic",), grid_body="pelvis")
    terrain = TerrainObservation(scene, cfg)
    np.testing.assert_array_equal(terrain._state.build_state(scene_data), expected)
    scene_data.qpos[-7:-4] = [-2, 4, 8]
    scene_data.qvel[-6:] = np.arange(6)
    mujoco.mj_forward(scene, scene_data)
    np.testing.assert_array_equal(terrain._state.build_state(scene_data), expected)


def test_scene_launch_preserves_actor_observation_timing() -> None:
    """Teleporting a separate prop must not refresh the actor's delayed features."""
    from myosuite.integrations.musclemimic.terrain_policy import TerrainController

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option gravity="0 0 0" timestep=".01"/>
      <worldbody><body name="actor"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body pos="4 0 0"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    observed = []

    class Policy:
        obs_adapter = None

        def reset(self) -> None:
            observed.clear()

        def action_for(
            self, state: mujoco.MjData, reference: object, frame: int
        ) -> np.ndarray:
            observed.append(float(state.body("actor").xpos[0]))
            return np.zeros(0)

    reference = SimpleNamespace(qpos=np.zeros((2, 7)), qvel=np.zeros((2, 6)))
    controller = TerrainController(Policy(), model, reference, 1)
    controller.reset()
    controller(data)
    mujoco.mj_step(model, data)
    data.qpos[7] = 6.0
    mujoco.mj_forward(model, data)
    assert data.body("actor").xpos[0] == pytest.approx(0.01)
    controller(data)
    assert observed[-1] == pytest.approx(0.0)
