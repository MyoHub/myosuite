# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.envs.heightfields import ChaseTagField, HeightField, TrackField
from myosuite.physics.quat_math import euler2mat, quat2euler
from myosuite.tests.test_envs import assert_close
from myosuite.utils.asset_path_resolver import resolve_model_xml_path
from myosuite import make_env

pytestmark = pytest.mark.tier2

_ASSETS = Path(__file__).parents[1] / "envs" / "myo" / "assets"


def _create_sim(xml_path: str):
    class Sim:
        def __init__(self, xml_path: str):
            resolved = resolve_model_xml_path(xml_path)
            self.model = mujoco.MjModel.from_xml_path(str(resolved))
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


def _reference_heightmap(field: HeightField) -> np.ndarray:
    """The heightmap with the yaw taken from the full ``quat2euler`` (the old path)."""
    qpos = field.mj_data.qpos
    rot_mat = euler2mat([0, 0, quat2euler(qpos[3:7])[2]])
    points = np.einsum("ij,kj->ik", field.height_points, rot_mat)
    points = points * field.view_distance + qpos[:3]
    px, py = field.cart2map(points[:, 1], points[:, 0])
    window = np.zeros((10, 10))
    window[:] = np.flipud(
        np.rot90(field.hfield.data[px, py].reshape(10, 10), axes=(1, 0))
    )
    return window.flatten()


@pytest.mark.parametrize("make_field", [_create_chasetagfield, _create_trackfield])
def test_heightmap_matches_full_euler_reference(
    make_field: Callable[[int], HeightField],
) -> None:
    """get_heightmap_obs is bit-identical to the quat2euler path at random poses."""
    field = make_field(0)
    field.sample()
    assert np.ptp(field.hfield.data) > 0
    rng = np.random.default_rng(0)
    qpos = field.mj_data.qpos
    # world y runs along the map rows (real_length), x along the columns.
    half_extent = 0.6 * np.array([field.real_width, field.real_length])
    varied = 0
    for i in range(300):
        qpos[:2] = rng.uniform(-half_extent, half_extent)
        quat = rng.normal(size=4)
        if i % 2:  # upright, yaw only
            quat[1:3] = 0.0
        qpos[3:7] = quat / np.linalg.norm(quat)
        heightmap = field.get_heightmap_obs()
        reference = _reference_heightmap(field)
        assert heightmap.dtype == reference.dtype and heightmap.shape == (100,)
        assert heightmap.tobytes() == reference.tobytes()
        varied += np.ptp(heightmap) > 0
    assert varied > 30


def test_quat2yaw_matches_quat2euler() -> None:
    """quat2yaw returns quat2euler(q)[2] bit for bit, edge cases included."""
    from myosuite.physics.quat_math import quat2yaw

    rng = np.random.default_rng(0)
    unit = rng.normal(size=(20000, 4))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    h = np.sqrt(0.5)
    gimbal = np.array([h, 0.0, h, 0.0]) + 1e-9 * rng.normal(size=(2000, 4))
    special = np.array(
        [
            [1, 0, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 0],
            [1e-9, 0, 0, 0],
            [h, 0, h, 0],
            [h, 0, -h, 0],
            [0.5, 0.5, 0.5, 0.5],
            [np.nan, 0, 0, 0],
        ]
    )
    quats = np.concatenate([unit, 3.7 * unit[:2000], gimbal, special])
    with np.errstate(divide="ignore", invalid="ignore"):
        for quat in quats:
            expected = quat2euler(quat)[2]
            yaw = quat2yaw(quat)
            assert type(yaw) is type(expected)
            assert np.asarray(yaw).tobytes() == np.asarray(expected).tobytes(), quat


@pytest.mark.parametrize(
    "env_id", ["myoChallengeOslRunFixed-v0", "myoChallengeChaseTagP2-v0"]
)
def test_obs_dict_keeps_unobserved_heightmap(env_id: str) -> None:
    """``info["obs_dict"]`` carries the heightmap although it is not observed.

    The PyPI obs-dict keys (test_challenge_pypi_regression) and the challenge
    docs include ``hfield``, so it is computed every step.
    """
    env = make_env(env_id)
    try:
        env.reset(seed=0)
        assert "hfield" not in env.unwrapped.obs_keys
        *_, info = env.step(np.zeros(env.action_space.shape, env.action_space.dtype))
        assert info["obs_dict"]["hfield"].shape == (100,)
    finally:
        env.close()
