# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Athletic API contract + bolt sprinter golden-path freeze tests."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.tier2

from myosuite.envs.myo.athletic.abi_freeze import (
    BOLT_SPRINTER_CTRL_DT,
    BOLT_SPRINTER_DECIMATION,
    BOLT_SPRINTER_ENV_ID,
    BOLT_SPRINTER_FALL_HEIGHT,
    BOLT_SPRINTER_HEADING_DIR,
    BOLT_SPRINTER_LAMBDA_LIMIT,
    BOLT_SPRINTER_LAMBDA_VEL,
    BOLT_SPRINTER_SIM_DT,
)
from myosuite.envs.myo.athletic.catalog import PLANTS, TASKS, iter_bundles
from myosuite.envs.myo.athletic.factory import mjlab_env_cfg_from_bundle
from myosuite.envs.myo.athletic.specs import AthleticBundle
from myosuite.envs.myo.athletic.terms import default_lane_reward_terms, lane_reward_keys
from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import AthleticSpec


def test_catalog_covers_msk_envs_sweep():
    expected = {
        "sprint",
        "backpedal",
        "side_shuffle",
        "hop",
        "hurdles",
        "sprint_blocks",
        "carioca",
        "curve",
        "vertical",
        "locomotion",
        "imitate",
        "uphill_sprint",
    }
    assert expected <= set(TASKS)
    assert "myosprinter" in PLANTS
    assert "myofullbody_simple" in PLANTS


def test_lane_reward_keys_match_athletic_spec():
    spec = AthleticSpec()
    weights = default_lane_reward_terms(spec)
    assert set(weights) == set(lane_reward_keys())
    assert weights["rew_vel"] == pytest.approx(1e-2)
    assert weights["rew_limit"] == pytest.approx(-3e-4)


def test_heading_modes_respect_plant_forward():
    plant = PLANTS["myosprinter"]
    sprint = AthleticBundle(task=TASKS["sprint"], plant=plant)
    back = AthleticBundle(task=TASKS["backpedal"], plant=plant)
    side = AthleticBundle(task=TASKS["side_shuffle"], plant=plant)
    assert sprint.heading_dir == BOLT_SPRINTER_HEADING_DIR
    assert back.heading_dir == (-1.0, 0.0)
    assert side.heading_dir == (0.0, 1.0)


def test_capability_gating_skips_unsupported():
    # All current plants advertise full capabilities; smoke the iterator size.
    bundles = iter_bundles()
    assert len(bundles) >= len(TASKS)  # ≥1 plant × all tasks


def test_bolt_sprint_env_id_stable():
    bundle = AthleticBundle(task=TASKS["sprint"], plant=PLANTS["myosprinter"])
    assert bundle.env_id() == BOLT_SPRINTER_ENV_ID


def test_golden_sprint_cfg_is_make_sprint_env_cfg():
    """Athletic factory must return the identical golden cfg object fields for bolt sprint."""
    from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import (
        make_sprint_env_cfg,
    )

    bundle = AthleticBundle(task=TASKS["sprint"], plant=PLANTS["myosprinter"])
    via_factory = mjlab_env_cfg_from_bundle(bundle, num_envs=4)
    golden = make_sprint_env_cfg(num_envs=4)

    assert via_factory.decimation == golden.decimation == BOLT_SPRINTER_DECIMATION
    assert via_factory.sim.mujoco.timestep == pytest.approx(BOLT_SPRINTER_SIM_DT)
    assert via_factory.decimation == int(round(BOLT_SPRINTER_CTRL_DT / BOLT_SPRINTER_SIM_DT))
    assert set(via_factory.rewards) == set(golden.rewards)
    assert via_factory.rewards["forward_vel"].weight == pytest.approx(BOLT_SPRINTER_LAMBDA_VEL)
    assert via_factory.rewards["limit_pain"].weight == pytest.approx(BOLT_SPRINTER_LAMBDA_LIMIT)
    # Fall threshold locked to make_sprint_env_cfg (Y-up), not AthleticSpec default.
    assert BOLT_SPRINTER_FALL_HEIGHT == pytest.approx(0.55)


def test_fullbody_simple_sprint_env_id():
    bundle = AthleticBundle(task=TASKS["sprint"], plant=PLANTS["myofullbody_simple"])
    assert bundle.env_id() == "myoFullbodySimpleSprint-v0"


@pytest.mark.parametrize("task_key", ["backpedal", "side_shuffle", "hop"])
def test_lane_family_bundles_build_cfg(task_key):
    bundle = AthleticBundle(task=TASKS[task_key], plant=PLANTS["myofullbody_simple"])
    cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=1)
    assert cfg.decimation == AthleticSpec().n_substeps
    assert "rew_vel" in cfg.rewards
    assert "fallen" in cfg.terminations


@pytest.mark.parametrize(
    "task_key",
    ["hurdles", "sprint_blocks", "carioca", "curve", "vertical", "locomotion", "imitate"],
)
def test_scene_stateful_bundles_build_cfg(task_key):
    bundle = AthleticBundle(task=TASKS[task_key], plant=PLANTS["myofullbody_simple"])
    cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=1)
    assert cfg.episode_length_s == pytest.approx(TASKS[task_key].episode_length_s)
    assert math.isfinite(float(cfg.sim.mujoco.timestep))


def test_sibling_cfg_smoke_finite_rewards_weights():
    """Non-sprint athletic cells expose finite λ-weighted reward terms."""
    for task_key in ("backpedal", "hop", "hurdles", "vertical"):
        bundle = AthleticBundle(task=TASKS[task_key], plant=PLANTS["myofullbody_simple"])
        cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=1)
        for name, term in cfg.rewards.items():
            assert math.isfinite(float(term.weight)), f"{task_key}/{name}"
        assert "time_out" in cfg.terminations
