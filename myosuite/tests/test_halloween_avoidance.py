"""Contracts for the opt-in TERRA curriculum; no checkpoint download required."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.myo.tasks.halloween.config import AvoidanceConfig

pytestmark = pytest.mark.tier1


@pytest.fixture
def env():
    instance = make_env("myoHalloween-v0", profile="terra", max_episode_steps=3000)
    yield instance
    instance.close()


def test_terra_reset_reproducible_and_no_network(env):
    obs, info = env.reset(seed=2027)
    qpos = env.unwrapped.data.qpos.copy()
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert obs.shape == (535,) and env.action_space.shape == (354,)
    assert info["halloween"]["launch_id"] == 0
    assert info["halloween"]["policy_seed"] == 1357624910
    assert np.linalg.norm(env.unwrapped.data.act) > 0
    env.step(np.zeros(354))
    obs2, info2 = env.reset(seed=2027)
    np.testing.assert_array_equal(obs, obs2)
    np.testing.assert_array_equal(qpos, env.unwrapped.data.qpos)
    assert info2["halloween"]["events"] == []
    assert env.unwrapped.avoidance.steps == 0
    assert env.unwrapped.model.body_gravcomp[env.unwrapped.avoidance.body] == 1


def test_launch_is_visible_in_observation_before_response(env):
    env.reset(seed=2027)
    task = env.unwrapped.avoidance
    # One step before the sampled launch: shorten the wait without changing physics.
    task.wait = 1
    obs, _, _, _, info = env.step(np.zeros(354))
    assert info["halloween"]["launch_id"] == 1
    assert info["halloween"]["steps_since_launch"] == 0
    assert obs[-6 + 1] < -5
    assert obs[-3 + 1] == 8.0
    assert env.unwrapped.data.qvel[task.va + 1] == 8.0
    assert env.unwrapped.model.body_gravcomp[task.body] == 0


def test_actual_body_contact_terminates(env):
    env.reset(seed=2027)
    task = env.unwrapped.avoidance
    model, data = env.unwrapped.model, env.unwrapped.data
    target = next(
        i for i in task.body_geoms if model.geom_bodyid[i] == model.body("head").id
    )
    data.qpos[task.qa : task.qa + 3] = data.geom_xpos[target]
    data.qvel[task.va : task.va + 6] = 0
    mujoco.mj_forward(model, data)
    _, _, terminated, _, info = env.step(np.zeros(354))
    assert terminated and info["rwd_dict"]["hit"]
    assert info["rwd_dict"]["body_contacts"] > 0
    assert info["halloween"]["current_event"]["body_contacts"] > 0


def test_info_does_not_mutate_task(env):
    _, info = env.reset(seed=2027)
    info["halloween"]["home"][:] = 999
    assert np.max(np.abs(env.unwrapped.avoidance.home)) < 1


@pytest.mark.parametrize("profile", ["benchmark", "terra"])
def test_constructor_seed_and_pickle_round_trip(profile):
    import pickle

    kwargs = dict(profile=profile, seed=1234, render_mode="rgb_array")
    if profile == "terra":
        kwargs["avoidance_config"] = replace(AvoidanceConfig(), between_wait_steps=60)
    with make_env("myoHalloween-v0", **kwargs) as original:
        assert original.unwrapped.get_input_seed() == 1234
        obs, _ = original.reset()
        action = np.full(354, 0.01, dtype=np.float32)
        expected = original.step(action)
        with pickle.loads(pickle.dumps(original)) as restored:
            assert restored.unwrapped.profile == profile
            assert restored.unwrapped.render_mode == "rgb_array"
            assert restored.unwrapped.get_input_seed() == 1234
            assert restored.action_space == original.action_space
            assert restored.observation_space == original.observation_space
            reset_obs, _ = restored.reset()
            np.testing.assert_array_equal(obs, reset_obs)
            actual = restored.step(action)
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            if profile == "terra":
                assert restored.unwrapped.avoidance.config == kwargs["avoidance_config"]


def test_tutorial_sources_match():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (root / "tutorials/4.5_Halloween_Avoidance.py").read_text()
    notebook = json.loads(
        (root / "tutorials/4.5_Halloween_Avoidance.ipynb").read_text()
    )
    script_cells = [
        chunk.strip()
        for chunk in source.split("# %%")[1:]
        if not chunk.startswith(" [markdown]")
    ]
    notebook_cells = [
        "".join(cell["source"]).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert notebook_cells[1:-1] == script_cells


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "unknown"},
        {"avoidance_config": AvoidanceConfig()},
        {"profile": "terra", "curriculum": True},
        {"profile": "terra", "avoidance_config": replace(AvoidanceConfig(), throws=())},
    ],
)
def test_invalid_profile_configuration(kwargs):
    with pytest.raises(ValueError):
        make_env("myoHalloween-v0", **kwargs)
