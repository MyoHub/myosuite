# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Athletic API contract + bolt sprinter golden-path freeze tests."""

from __future__ import annotations

import math

import pytest

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
from myosuite.envs.myo.athletic.bodies import (
    FULLBODY_BODY_MAP,
    FULLBODY_SIMPLE_BODY_MAP,
    SPRINTER_BODY_MAP,
)
from myosuite.envs.myo.athletic.catalog import (
    PLANTS,
    TASKS,
    athletic_bundle,
    iter_bundles,
)
from myosuite.envs.myo.athletic.factory import (
    AthleticModularTaskEnv,
    mjlab_env_cfg_from_bundle,
    task_config_from_bundle,
)
from myosuite.envs.myo.athletic.specs import (
    DEFAULT_MODEL,
    DEFAULT_TENDONS,
    AthleticBundle,
)
from myosuite.envs.myo.athletic.terms import default_lane_reward_terms, lane_reward_keys
from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import (
    ELASTIC_N_SUBSTEPS,
    ELASTIC_SIM_DT,
    AthleticSpec,
    SPRINT_SIM_DT,
)

pytestmark = pytest.mark.tier2


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
    assert "myofullbody" in PLANTS
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
    # All current plants advertise full capabilities; rigid + elastic per cell.
    bundles = iter_bundles()
    assert len(bundles) >= 2 * len(TASKS)  # ≥1 plant × all tasks × 2 tendons
    rigid_only = iter_bundles(tendons=("rigid",))
    assert all(b.tendons == "rigid" for b in rigid_only)


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
    assert via_factory.decimation == int(
        round(BOLT_SPRINTER_CTRL_DT / BOLT_SPRINTER_SIM_DT)
    )
    assert set(via_factory.rewards) == set(golden.rewards)
    assert via_factory.rewards["forward_vel"].weight == pytest.approx(
        BOLT_SPRINTER_LAMBDA_VEL
    )
    assert via_factory.rewards["limit_pain"].weight == pytest.approx(
        BOLT_SPRINTER_LAMBDA_LIMIT
    )
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
    [
        "hurdles",
        "sprint_blocks",
        "carioca",
        "curve",
        "vertical",
        "locomotion",
        "imitate",
    ],
)
def test_scene_stateful_bundles_build_cfg(task_key):
    bundle = AthleticBundle(task=TASKS[task_key], plant=PLANTS["myofullbody_simple"])
    cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=1)
    assert cfg.episode_length_s == pytest.approx(TASKS[task_key].episode_length_s)
    assert math.isfinite(float(cfg.sim.mujoco.timestep))


def test_sibling_cfg_smoke_finite_rewards_weights():
    """Non-sprint athletic cells expose finite λ-weighted reward terms."""
    for task_key in ("backpedal", "hop", "hurdles", "vertical"):
        bundle = AthleticBundle(
            task=TASKS[task_key], plant=PLANTS["myofullbody_simple"]
        )
        cfg = mjlab_env_cfg_from_bundle(bundle, num_envs=1)
        for name, term in cfg.rewards.items():
            assert math.isfinite(float(term.weight)), f"{task_key}/{name}"
        assert "time_out" in cfg.terminations


def test_default_model_is_myofullbody_with_rigid_tendons():
    bundle = athletic_bundle("sprint")
    assert DEFAULT_MODEL == "myofullbody"
    assert DEFAULT_TENDONS == "rigid"
    assert bundle.plant.key == "myofullbody"
    assert bundle.tendons == "rigid"
    assert bundle.action_kind == "native_muscle"
    assert bundle.env_id() == "myoFullbodySprint-v0"
    assert bundle.athletic.sim_dt == pytest.approx(SPRINT_SIM_DT)


@pytest.mark.parametrize(
    ("model", "tendons", "action", "env_id"),
    [
        ("myofullbody", "rigid", "native_muscle", "myoFullbodySprint-v0"),
        ("myofullbody", "elastic", "bolt_muscle", "myoFullbodySprintElastic-v0"),
        ("myofullbody_simple", "rigid", "bolt_muscle", "myoFullbodySimpleSprint-v0"),
        (
            "myofullbody_simple",
            "elastic",
            "bolt_muscle",
            "myoFullbodySimpleSprintElastic-v0",
        ),
        ("myosprinter", "rigid", "bolt_muscle", "myoSprint-v0"),
        ("myosprinter", "elastic", "bolt_muscle", "myoSprintElastic-v0"),
    ],
)
def test_model_and_tendons_select_actuator_and_env_id(model, tendons, action, env_id):
    bundle = athletic_bundle("sprint", model, tendons)
    assert bundle.action_kind == action
    assert bundle.env_id() == env_id


def test_elastic_variant_declares_its_own_physics():
    rigid = athletic_bundle("sprint")
    elastic = athletic_bundle("sprint", tendons="elastic")
    assert rigid.env_id() != elastic.env_id()
    assert rigid.athletic.sim_dt == pytest.approx(SPRINT_SIM_DT)
    assert elastic.athletic.sim_dt == pytest.approx(ELASTIC_SIM_DT)
    assert elastic.athletic.n_substeps == ELASTIC_N_SUBSTEPS
    assert elastic.athletic.ctrl_dt == pytest.approx(rigid.athletic.ctrl_dt)
    cfg = task_config_from_bundle(elastic)
    assert cfg.backend.sim_dt == pytest.approx(ELASTIC_SIM_DT)
    assert cfg.backend.n_substeps == ELASTIC_N_SUBSTEPS


def test_unknown_tendons_rejected():
    with pytest.raises(ValueError, match="tendons"):
        athletic_bundle("sprint", tendons="stiff")


def test_task_config_records_model_and_tendons():
    cfg = task_config_from_bundle(
        athletic_bundle("sprint", "myofullbody_simple", "elastic")
    )
    assert cfg.model == "myofullbody_simple"
    assert cfg.reward.extra["model"] == "myofullbody_simple"
    assert cfg.reward.extra["tendons"] == "elastic"


def test_elastic_is_not_built_as_a_rigid_env():
    bundle = athletic_bundle("sprint", tendons="elastic")
    with pytest.raises(NotImplementedError, match="elastic"):
        mjlab_env_cfg_from_bundle(bundle, num_envs=1)
    sprinter = athletic_bundle("sprint", "myosprinter", "elastic")
    with pytest.raises(NotImplementedError, match="elastic"):
        mjlab_env_cfg_from_bundle(sprinter, num_envs=1)


def test_make_env_kwargs_keep_the_registered_skeleton_and_tendons():
    cfg = task_config_from_bundle(athletic_bundle("sprint"))
    with pytest.raises(ValueError, match="Elastic"):
        AthleticModularTaskEnv(cfg, tendons="elastic")
    with pytest.raises(ValueError, match="myofullbody_simple"):
        AthleticModularTaskEnv(cfg, model="myofullbody_simple")
    elastic_cfg = task_config_from_bundle(athletic_bundle("sprint", tendons="elastic"))
    with pytest.raises(NotImplementedError, match="elastic"):
        AthleticModularTaskEnv(elastic_cfg)


def test_body_maps_expose_names_not_indices():
    for body_map in (SPRINTER_BODY_MAP, FULLBODY_SIMPLE_BODY_MAP, FULLBODY_BODY_MAP):
        for logical in ("root", "toes_l", "toes_r", "hand_r", "knee_l", "knee_r"):
            name = body_map.resolve(logical)
            assert isinstance(name, str) and name
            assert not name.isdigit()
    assert PLANTS["myofullbody"].body_map is FULLBODY_BODY_MAP
