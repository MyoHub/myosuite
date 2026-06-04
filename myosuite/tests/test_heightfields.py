# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import gymnasium as gym
import mujoco
import pytest

from myosuite.envs.heightfields import ChaseTagField, TrackField
from myosuite.tests.test_envs import assert_close

pytestmark = pytest.mark.tier2

_ASSETS = Path(__file__).parents[1] / "envs" / "myo" / "assets"


def _create_sim(xml_path: str):
    class Sim:
        def __init__(self, xml_path: str):
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)

    return Sim(xml_path)


def _create_chasetagfield(seed: int) -> ChaseTagField:
    np_random = gym.utils.seeding.np_random(seed)[0]
    xml_path = str(_ASSETS / "leg" / "myolegs_chasetag.xml")
    sim = _create_sim(xml_path)
    return ChaseTagField(
        mj_model=sim.model,
        mj_data=sim.data,
        rng=np_random,
        rough_range=(0.0, 0.05),
        hills_range=(0.0, 0.1),
        relief_range=(0.0, 0.05),
    )


def _create_trackfield(seed: int) -> TrackField:
    np_random = gym.utils.seeding.np_random(seed)[0]
    xml_path = str(_ASSETS / "leg" / "myoosl_runtrack.xml")
    sim = _create_sim(xml_path)
    return TrackField(
        rough_difficulties=[0.0, 0.1, 0.2],
        hills_difficulties=[0.0, 0.1, 0.2],
        stairs_difficulties=[0.0, 0.1, 0.2],
        mj_model=sim.model,
        mj_data=sim.data,
        rng=np_random,
    )


def test_chasetagfield() -> None:
    seed = 42
    heightfield = _create_chasetagfield(seed)
    heightfield.sample()
    data = heightfield.hfield.data.copy()
    heightfield2 = _create_chasetagfield(seed)
    heightfield2.sample()
    data2 = heightfield2.hfield.data.copy()
    assert_close(data, data2)


def test_trackfield() -> None:
    seed = 42
    heightfield = _create_trackfield(seed)
    heightfield.sample()
    data = heightfield.hfield.data.copy()
    heightfield2 = _create_trackfield(seed)
    heightfield2.sample()
    data2 = heightfield2.hfield.data.copy()
    assert_close(data, data2)
