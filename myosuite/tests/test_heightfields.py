import unittest
import os
import gymnasium as gym
import mujoco
from myosuite.envs.heightfields import ChaseTagField, TrackField
from myosuite.tests.test_envs import assert_close


class TestHeightfields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(os.path.abspath(__file__))

    def _load_model(self, xml_path):
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        return mj_model, mujoco.MjData(mj_model)

    def _create_chasetagfield(self, seed):
        np_random = gym.utils.seeding.np_random(seed)[0]
        xml_path = os.path.join(self.curr_dir, "../envs/myo/assets/leg/myolegs_chasetag.xml")
        mj_model, mj_data = self._load_model(xml_path)
        return ChaseTagField(
            mj_model=mj_model,
            mj_data=mj_data,
            rng=np_random,
            rough_range=(0.0, 0.1),
            hills_range=(0.0, 0.1),
            relief_range=(0.0, 0.1),
            )

    def _create_trackfield(self, seed):
        np_random = gym.utils.seeding.np_random(seed)[0]
        xml_path = os.path.join(self.curr_dir, "../envs/myo/assets/leg/myoosl_runtrack.xml")
        mj_model, mj_data = self._load_model(xml_path)
        return TrackField(
            mj_model=mj_model,
            mj_data=mj_data,
            rng=np_random,
            rough_difficulties=[0.0, 0.1, 0.2],
            hills_difficulties=[0.0, 0.1, 0.2],
            stairs_difficulties=[0.0, 0.1, 0.2],
            )

    def test_chasetagfield(self):
        seed = 42
        heightfield = self._create_chasetagfield(seed)
        heightfield.sample()
        data = heightfield.hfield.data.copy()
        heightfield2 = self._create_chasetagfield(seed)
        heightfield2.sample()
        data2 = heightfield2.hfield.data.copy()
        assert_close(data, data2)

    def test_trackfield(self):
        seed = 42
        heightfield = self._create_trackfield(seed)
        heightfield.sample()
        data = heightfield.hfield.data.copy()
        heightfield2 = self._create_trackfield(seed)
        heightfield2.sample()
        data2 = heightfield2.hfield.data.copy()
        assert_close(data, data2)


if __name__ == '__main__':
    unittest.main()
