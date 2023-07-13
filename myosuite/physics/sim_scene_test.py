""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Unit tests for SimScene."""

import contextlib
import tempfile
from typing import Generator

from absl.testing import absltest

from myosuite.physics.sim_scene import SimBackend, SimScene

# Simple MuJoCo model XML.
TEST_MODEL_XML = """
<mujoco model="test">
    <compiler coordinate="global"/>
    <worldbody>
        <light pos="0 1 1" dir="0 -1 -1" diffuse="1 1 1"/>
        <body name="main">
            <geom name="base" type="capsule" fromto="0 0 1  0 0 0.6" size="0.06"/>
            <body>
                <geom type="capsule" fromto="0 0 0.6  0.3 0 0.6" size="0.04"/>
                <joint name="j1" type="hinge" pos="0 0 0.6" axis="0 1 0"/>
                <joint name="j2" type="hinge" pos="0 0 0.6" axis="1 0 0"/>
                <body>
                    <geom type="ellipsoid" pos="0.4 0 0.6" size="0.1 0.08 0.02"/>
                    <site name="end" pos="0.5 0 0.6" type="sphere" size="0.01"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 1 0"/>
                    <joint type="hinge" pos="0.3 0 0.6" axis="0 0 1"/>
                </body>
            </body>
        </body>
    </worldbody>
</mujoco>
"""


@contextlib.contextmanager
def test_model_file() -> Generator[str, None, None]:
    """Context manager that yields a temporary MuJoCo XML file."""
    with tempfile.NamedTemporaryFile(mode='w+t', suffix='.xml') as f:
        f.write(TEST_MODEL_XML)
        f.flush()
        f.seek(0)
        yield f.name


def mjpy_and_dm(fn):
    """Decorator that tests for both mujoco_py and dm_control."""

    def test_fn(self: absltest.TestCase):
        with test_model_file() as test_file_path:
            with self.subTest('mujoco_py'):
                fn(
                    self,
                    SimScene.create(
                        test_file_path, backend=SimBackend.MUJOCO_PY))
            with self.subTest('dm_control'):
                fn(
                    self,
                    SimScene.create(
                        test_file_path, backend=SimBackend.MUJOCO))

    return test_fn


class SimSceneTest(absltest.TestCase):
    """Unit test class for SimScene."""

    @mjpy_and_dm
    def test_load(self, robot: SimScene):
        self.assertIsNotNone(robot.sim)
        self.assertIsNotNone(robot.model)
        self.assertIsNotNone(robot.data)

    @mjpy_and_dm
    def test_step(self, robot: SimScene):
        robot.sim.reset()
        robot.sim.forward()
        robot.sim.step()

        robot.renderer.render_to_window()
        for _ in range(10):
            # robot.sim.step()
            robot.data.qpos[0] += 0.01
            robot.advance()
            # input()

    @mjpy_and_dm
    def test_accessors(self, robot: SimScene):
        self.assertTrue(robot.model.body_name2id('main') >= 0)
        self.assertTrue(robot.model.geom_name2id('base') >= 0)
        self.assertTrue(robot.model.site_name2id('end') >= 0)
        self.assertTrue(robot.model.joint_name2id('j1') >= 0)
        self.assertIsNotNone(robot.data.body_xpos[0])
        self.assertIsNotNone(robot.data.body_xquat[0])

    @mjpy_and_dm
    def test_copy_model(self, robot: SimScene):
        initial_pos = robot.model.body_pos[0].copy().tolist()

        model_copy = robot.copy_model()
        robot.model.body_pos[0, :] = [0.1, 0.2, 0.3]

        self.assertListEqual(model_copy.body_pos[0].tolist(), initial_pos)
        self.assertListEqual(robot.model.body_pos[0].tolist(), [0.1, 0.2, 0.3])


if __name__ == '__main__':
    absltest.main()
